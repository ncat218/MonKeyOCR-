"""Bootstrap the Phase 1 verified stack automatically, scoped to the datastore workspace."""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from .common import Processes, identity, locked_json, read_json, save_json, sha256, utc_stamp, stop_signals


def launch(repo, config_path, workspace, dataset_parent):
    config = read_json(config_path)
    root = workspace / config['run_name']
    root.mkdir(parents=True, exist_ok=True)
    # Prevent two overlapping allocations from writing the same experiment suite.
    import fcntl
    with (root / 'run_all__exclusive_process.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('This run_name already has a live controller. Do not submit duplicate GPU jobs.') from exc
        cache = workspace / 'environment_and_models__phase1_verified_stack'
        cache.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env.update(XDG_CACHE_HOME=str(cache/'cache'), HF_HOME=str(cache/'huggingface_cache'),
                   UV_CACHE_DIR=str(cache/'uv_cache'), UV_PYTHON_INSTALL_DIR=str(cache/'python_installations'),
                   PIP_CACHE_DIR=str(cache/'pip_cache'), TORCH_EXTENSIONS_DIR=str(cache/'torch_extensions'),
                   PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_NO_INPUT='1', PYTHONUNBUFFERED='1')
        logs = root/'logs'/('allocation_' + os.environ.get('SLURM_JOB_ID', 'direct') + '__' + utc_stamp())
        proc = Processes(logs, env)
        for sig in stop_signals():
            signal.signal(sig, proc.request_stop)
        # Fail on missing/ambiguous uploaded datasets before spending allocation time on CUDA compilation.
        from .data import discover_inputs
        locations=discover_inputs(dataset_parent,config['data'],exclude=[workspace,repo])
        save_json(logs/'00_early_dataset_location_check__before_gpu_stack_install.json',locations)
        if not shutil.which('nvidia-smi'):
            raise RuntimeError('nvidia-smi unavailable: request a GPU compute allocation before setup.')
        proc.run(['nvidia-smi'], '00_h200_mig_hardware_inventory')
        cuda = find_cuda(config['environment']['cuda_home'], config['environment']['allow_recorded_cuda12_minor_mismatch'])
        env.update(CUDA_HOME=cuda['home'], PATH=cuda['home']+'/bin'+os.pathsep+env['PATH'],
                   MAX_JOBS=str(config['environment']['build_jobs']), MPLBACKEND='Agg',
                   TOKENIZERS_PARALLELISM='false', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
        save_json(logs/'00_selected_cuda_toolkit.json', cuda)
        py = ensure_python311(cache, proc, env)
        env.update(VIRTUAL_ENV=str(py.parent.parent), PATH=str(py.parent)+os.pathsep+env['PATH'],
                   PIP_CONSTRAINT=str(repo/'constraints-gpu.txt'))
        proc.env = env
        locks = read_json(repo/'configs/verified_phase1_phase2_sources.json')
        specification = dict(lock=locks, constraints=sha256(repo/'constraints-gpu.txt'),
                             cuda=cuda, bootstrap=sha256(__file__), python='3.11')
        marker = cache/'stack_installed_and_probed.json'
        if marker.exists() and read_json(marker)['specification'] != specification:
            raise RuntimeError('Cached stack specification changed. Use a separate workspace, not a mixed environment.')
        monkey = cache/'upstream_MonkeyOCRv2'
        addon = monkey/'detection/mmocr'
        mmocr = addon/'mmocr'
        if not marker.exists():
            checkout(proc, 'https://github.com/Yuliang-Liu/MonkeyOCRv2.git', monkey, locks['monkey_commit'], 'monkey')
            proc.run([py,'-m','pip','install','torch==2.9.0','torchvision==0.24.0',
                      '--index-url','https://download.pytorch.org/whl/cu128'], '02_pytorch_cu128')
            proc.run([py,'-m','pip','install','numpy==1.26.4','setuptools<81','wheel','ninja',
                      'mmengine==0.10.7','opencv-python==4.10.0.84'], '03_core_build_dependencies')
            probe_log = proc.run([py,'-c',
                'import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count()==1, "One MIG GPU per job required"; '
                'print("ARCH="+".".join(map(str,torch.cuda.get_device_capability(0)))); '
                'print(torch.cuda.get_device_name(0),torch.cuda.get_device_properties(0).total_memory)'], '04_gpu_architecture')
            arch = re.search(r'^ARCH=(\d+\.\d+)', probe_log.read_text(), re.M).group(1)
            env['TORCH_CUDA_ARCH_LIST'] = arch
            mmcv = cache/'upstream_mmcv_2_0_1'
            checkout(proc,'https://github.com/open-mmlab/mmcv.git',mmcv,locks['mmcv_commit'],'mmcv')
            original = subprocess.check_output(['git','show','HEAD:setup.py'],cwd=mmcv,text=True,env=env)
            (mmcv/'setup.py').write_text(original.replace('-std=c++14','-std=c++17'),encoding='utf-8')
            proc.run([py,'-m','pip','install','-v','-e',mmcv,'--no-build-isolation'],
                     '05_mmcv_cuda_build_h200_arch_'+arch.replace('.','_'),
                     env=dict(env,MMCV_WITH_OPS='1',FORCE_CUDA='1'))
            # Pre-create the exact nested checkout so the official installer cannot float to a different tag.
            checkout(proc,'https://github.com/open-mmlab/mmocr.git',mmocr,locks['mmocr_commit'],'mmocr')
            proc.run(['bash','install.sh'], '06_official_monkey_detection_addon_install', cwd=addon)
            proc.run([py,'-m','pip','install','mmdet==3.1.0','transformers==4.57.1','safetensors==0.7.0',
                      'huggingface_hub','einops','timm==1.0.30','Pillow==13.0.0','scipy','shapely==2.2.0',
                      'pyclipper','imgaug==0.4.0','ftfy','lmdb','tensorboard','future'], '07_detection_training_dependencies')
            proc.run([py,'-m','pip','install','-e',mmocr], '08_editable_mmocr')
            proc.run(['git','diff'], '08_installed_mmocr_addon_source_diff',cwd=mmocr)
            env['PYTHONPATH'] = str(repo/'src')+os.pathsep+str(mmocr)
            proc.run([py,'-m','monkey_textdet.gpu_checks','--download','--mmocr-root',mmocr,
                      '--lock',repo/'configs/verified_phase1_phase2_sources.json',
                      '--output',logs/'09_gpu_ops_and_model_artifacts.json'], '09_download_and_cuda_forward_backward_probe', cwd=mmocr)
            save_json(marker, dict(specification=specification, architecture=arch, completed_utc=utc_stamp()))
        env['PYTHONPATH'] = str(repo/'src')+os.pathsep+str(mmocr)
        proc.run([py,'-m','monkey_textdet.gpu_checks','--mmocr-root',mmocr,
                  '--lock',repo/'configs/verified_phase1_phase2_sources.json',
                  '--output',logs/'10_revalidated_gpu_ops_and_model_artifacts.json'], '10_verify_cached_gpu_stack', cwd=mmocr)
        proc.run([py,'-m','pip','freeze'], '11_complete_environment_versions')
        command = [py,'-m','monkey_textdet.pipeline','--repo',repo,'--config',config_path,
                   '--workspace',workspace,'--mmocr-root',mmocr,'--logs',logs]
        if dataset_parent:
            command += ['--dataset-parent',dataset_parent]
        proc.run(command,'12_complete_experiment_suite',cwd=mmocr)


def find_cuda(explicit='', allow_minor=True):
    candidates = [Path(explicit)/'bin/nvcc'] if explicit else list(Path('/usr/local').glob('cuda*/bin/nvcc'))
    if not explicit and shutil.which('nvcc'):
        candidates.append(Path(shutil.which('nvcc')))
    found = {}
    for path in candidates:
        if path.is_file():
            text = subprocess.check_output([str(path),'--version'],text=True)
            match = re.search(r'release (\d+)\.(\d+)',text)
            if match and int(match[1])==12:
                key = str(path.resolve().parent.parent)
                found[key] = dict(home=key, version=match[1]+'.'+match[2], minor=int(match[2]))
    exact = [v for v in found.values() if v['version']=='12.8']
    choices = exact or (sorted(found.values(), key=lambda v:abs(v['minor']-8)) if allow_minor else [])
    if not choices:
        raise RuntimeError('No compatible CUDA 12.x nvcc toolkit found (12.8 preferred). '
                           'Load the HPC CUDA module or set environment.cuda_home; nvidia-smi driver CUDA is not nvcc.')
    chosen = choices[0]
    return dict(home=chosen['home'], version=chosen['version'], adapted_from_phase1_cu128=chosen['version']!='12.8')


def ensure_python311(cache, proc, env):
    py = cache/'python311_venv/bin/python'
    if py.is_file():
        return py
    host311 = sys.executable if sys.version_info[:2]==(3,11) else shutil.which('python3.11')
    if host311:
        proc.run([host311,'-m','venv',py.parent.parent], '01_create_python311_venv', env=env)
    else:
        uv = cache/'uv_bootstrap'
        if not (uv/'uv').is_dir():
            proc.run([sys.executable,'-m','pip','install','--target',uv,'uv'], '01_install_uv_bootstrap', env=env)
        uv_env = dict(env,PYTHONPATH=str(uv))
        proc.run([sys.executable,'-m','uv','python','install','3.11'], '01_download_python311', env=uv_env)
        proc.run([sys.executable,'-m','uv','venv','--python','3.11','--seed',py.parent.parent], '01_create_python311_venv', env=uv_env)
    return py


def checkout(proc,url,path,commit,label):
    if not path.exists():
        proc.run(['git','-c','core.autocrlf=false','clone','--filter=blob:none',url,path], 'source_clone_'+label)
    if not (path/'.git').is_dir():
        raise RuntimeError(f'Not a managed Git checkout: {path}')
    subprocess.run(['git','config','core.autocrlf','false'],cwd=path,check=True)
    current = subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()
    if current != commit:
        if subprocess.check_output(['git','status','--porcelain'],cwd=path,text=True).strip():
            raise RuntimeError(f'Dirty different source revision at {path}; will not overwrite it.')
        proc.run(['git','checkout','--detach',commit],'source_pin_'+label,cwd=path)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()==commit
