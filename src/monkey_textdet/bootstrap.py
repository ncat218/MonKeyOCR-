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
        cuda = ensure_cuda(cache,proc,config['environment'],env)
        env.update(CUDA_HOME=cuda['home'], PATH=cuda['home']+'/bin'+os.pathsep+env['PATH'],
                   MAX_JOBS=str(config['environment']['build_jobs']), MPLBACKEND='Agg',
                   TOKENIZERS_PARALLELISM='false', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
        # Conda toolkit headers/libraries may be under targets/<arch>-linux rather than root/include/lib64.
        env.update(cuda_build_paths(cuda['home'],env))
        save_json(logs/'00_selected_cuda_toolkit.json', cuda)
        compiler_probe=logs/'00_cuda_toolkit_compile_probe.cu'
        compiler_probe.write_text('#include <cuda_runtime.h>\n__global__ void toolkit_probe(float* x) { x[0] = 1.f; }\n',encoding='utf-8')
        proc.run([Path(cuda['home'])/'bin/nvcc','--std=c++17','-c',compiler_probe,
                  '-o',logs/'00_cuda_toolkit_compile_probe.o'],
                 '00_cuda_toolkit_and_host_compiler_compile_probe',env=env)
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


class CudaToolkitUnavailable(RuntimeError):
    def __init__(self,report):
        self.report=report
        detected=', '.join(p['version'] for p in report['detected_toolkits']) or 'none'
        super().__init__('No compatible CUDA 12.x nvcc toolkit found (12.8 preferred). '
                         'Detected toolkit versions: '+detected+'. Driver CUDA Version is not the toolkit version.')


def find_cuda(explicit='', allow_minor=True, extra_homes=()):
    if explicit:
        candidates=[Path(explicit).expanduser()/'bin/nvcc']
    else:
        homes=[*extra_homes,*[os.environ[k] for k in ('CUDA_HOME','CUDA_PATH','CONDA_PREFIX') if os.environ.get(k)]]
        candidates=[Path(home).expanduser()/'bin/nvcc' for home in homes]
        for base in ('/usr/local','/opt','/usr/lib'):
            candidates.extend(Path(base).glob('cuda*/bin/nvcc'))
        for base in ('/cm/shared/apps','/cm/local/apps'):
            candidates.extend(Path(base).glob('cuda*/bin/nvcc'))
            candidates.extend(Path(base).glob('cuda*/*/bin/nvcc'))
        candidates.extend(Path('/opt/nvidia/hpc_sdk').glob('Linux_x86_64/*/cuda/*/bin/nvcc'))
        if shutil.which('nvcc'):candidates.append(Path(shutil.which('nvcc')))
    candidates=list(dict.fromkeys(str(p.absolute()) for p in candidates))
    found = {};errors=[]
    for path in candidates:
        path=Path(path)
        if path.is_file():
            try:
                text = subprocess.check_output([str(path),'--version'],text=True,stderr=subprocess.STDOUT,timeout=15)
            except (OSError,subprocess.SubprocessError) as exc:
                errors.append(dict(nvcc=str(path),error=str(exc)));continue
            match = re.search(r'release (\d+)\.(\d+)',text)
            if match:
                # Resolve the prefix, not the nvcc symlink: Conda may link the binary into a separate package cache.
                key = str(path.parent.parent.resolve())
                found[key] = dict(home=key, version=match[1]+'.'+match[2], minor=int(match[2]))
    compatible=[v for v in found.values() if v['version'].startswith('12.')]
    exact = [v for v in compatible if v['version']=='12.8']
    choices = exact or (sorted(compatible, key=lambda v:abs(v['minor']-8)) if allow_minor else [])
    if not choices:
        raise CudaToolkitUnavailable(dict(searched_nvcc_paths=candidates,detected_toolkits=list(found.values()),probe_errors=errors))
    chosen = choices[0]
    return dict(home=chosen['home'], version=chosen['version'], adapted_from_phase1_cu128=chosen['version']!='12.8')


def find_conda():
    candidates=[os.environ.get('CONDA_EXE'),shutil.which('conda'),
                str(Path(os.environ['CONDA_PREFIX'])/'bin/conda') if os.environ.get('CONDA_PREFIX') else None,
                str(Path(sys.executable).parent/'conda')]
    return next((str(Path(p).resolve()) for p in candidates if p and Path(p).is_file()),None)


def private_toolkit_complete(prefix):
    prefix=Path(prefix)
    includes=[prefix/'include',*prefix.glob('targets/*/include')]
    libraries=[prefix/'lib64',prefix/'lib',*prefix.glob('targets/*/lib')]
    return ((prefix/'bin/nvcc').is_file() and (prefix/'conda-meta/history').is_file()
            and any((p/'cuda_runtime.h').is_file() for p in includes)
            and any((p/'libcudart.so').is_file() for p in libraries))


def ensure_cuda(cache,proc,options,env):
    private=cache/'cuda_toolkit_12_8_1__private_nvidia_conda_environment'
    extras=[private] if private_toolkit_complete(private) else []
    explicit=options.get('cuda_home','')
    allow_minor=options.get('allow_recorded_cuda12_minor_mismatch',True)
    try:
        return find_cuda(explicit,allow_minor,extra_homes=extras)
    except CudaToolkitUnavailable as exc:
        save_json(proc.log_dir/'00_cuda_toolkit_discovery__before_private_install.json',exc.report)
        if explicit or not options.get('auto_install_private_cuda128_if_missing',False):raise
    conda=find_conda()
    if not conda:
        raise RuntimeError('No usable CUDA 12.x toolkit and no Conda executable found. '
                           'Provide a toolkit via environment.cuda_home or make the HPC Conda executable available.')
    conda_env=dict(env,CONDA_PKGS_DIRS=str(cache/'conda_packages__private_cuda128'))
    action='install' if (private/'conda-meta/history').is_file() else 'create'
    proc.run([conda,action,'--yes','--prefix',private,'--override-channels','--strict-channel-priority',
              '--channel','nvidia/label/cuda-12.8.1','--channel','conda-forge','cuda-toolkit=12.8.1'],
             '00_install_private_cuda_toolkit_12_8_1__no_driver_or_base_changes',env=conda_env)
    if not private_toolkit_complete(private):
        raise RuntimeError('Private CUDA install lacks nvcc, headers or libcudart development library; inspect the install log.')
    result=find_cuda(str(private),allow_minor=False)
    save_json(cache/'private_cuda_toolkit_12_8_1__installation_receipt.json',
              dict(cuda=result,conda_executable=conda,channel='nvidia/label/cuda-12.8.1',package='cuda-toolkit=12.8.1'))
    return result


def cuda_build_paths(home,env):
    home=Path(home)
    includes=[p for p in [home/'include',*home.glob('targets/*/include')] if p.is_dir()]
    libraries=[p for p in [home/'lib64',home/'lib',*home.glob('targets/*/lib')] if p.is_dir()]
    result={}
    for key,paths in [('CPATH',includes),('CPLUS_INCLUDE_PATH',includes),('LIBRARY_PATH',libraries)]:
        prior=[p for p in env.get(key,'').split(os.pathsep) if p]
        result[key]=os.pathsep.join(dict.fromkeys([*[str(p) for p in paths],*prior]))
    # Do not prepend toolkit/stub paths to runtime LD_LIBRARY_PATH or override Torch's bundled CUDA libraries.
    return result


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
