"""CPU-only discovery/provisioning contract tests; never install or run CUDA in tests."""
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from test_core import tmp
from monkey_textdet.bootstrap import (CudaToolkitUnavailable,cuda_build_paths,ensure_cuda,
                                      find_cuda,private_toolkit_complete)


def toolkit(root):
    for name in ['bin/nvcc','conda-meta/history','targets/x86_64-linux/include/cuda_runtime.h',
                 'targets/x86_64-linux/lib/libcudart.so']:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('test fixture only')


class CudaBootstrapTests(unittest.TestCase):
    def test_honors_exported_cuda_home_outside_usr_local(self):
        with tmp() as d:
            root=Path(d)/'site-specific CUDA location';toolkit(root)
            with patch.dict(os.environ,{'CUDA_HOME':str(root)},clear=True),\
                 patch('shutil.which',return_value=None),patch('subprocess.check_output',return_value='Cuda compilation tools, release 12.8, V12.8.93'):
                self.assertEqual(find_cuda()['home'],str(root.resolve()))

    def test_rejects_cuda13_compiler_even_if_driver_supports_it(self):
        with tmp() as d:
            root=Path(d);toolkit(root)
            with patch('subprocess.check_output',return_value='Cuda compilation tools, release 13.0, V13.0.88'):
                with self.assertRaises(CudaToolkitUnavailable) as error:find_cuda(str(root))
            self.assertEqual(error.exception.report['detected_toolkits'][0]['version'],'13.0')

    def test_invalid_explicit_home_does_not_silently_use_other_home(self):
        with tmp() as d:
            root=Path(d);toolkit(root/'valid')
            with patch.dict(os.environ,{'CUDA_HOME':str(root/'valid')},clear=True):
                with self.assertRaises(CudaToolkitUnavailable):find_cuda(str(root/'not-installed'))

    def test_prefers_exact_128_over_other_cuda12_minor(self):
        with tmp() as d:
            root=Path(d);a=root/'cuda129';b=root/'cuda128';toolkit(a);toolkit(b)
            def version(command,**kwargs):return 'release 12.8,' if 'cuda128' in command[0] else 'release 12.9,'
            with patch.dict(os.environ,{},clear=True),patch('shutil.which',return_value=None),\
                 patch('subprocess.check_output',side_effect=version):
                self.assertEqual(find_cuda(extra_homes=[a,b])['home'],str(b.resolve()))

    def test_conda_target_include_and_library_paths_preserve_existing_paths(self):
        with tmp() as d:
            root=Path(d);toolkit(root)
            paths=cuda_build_paths(root,dict(CPATH='/existing/include',LIBRARY_PATH='/existing/lib',LD_LIBRARY_PATH='/torch/runtime'))
            self.assertIn(str(root/'targets/x86_64-linux/include'),paths['CPATH'])
            self.assertIn('/existing/include',paths['CPATH'])
            self.assertIn(str(root/'targets/x86_64-linux/lib'),paths['LIBRARY_PATH'])
            self.assertNotIn('LD_LIBRARY_PATH',paths)

    def test_missing_toolkit_provisions_private_prefix_not_base_environment(self):
        with tmp() as d:
            cache=Path(d);calls=[]
            class Process:
                log_dir=cache/'logs'
                def run(self,command,label,env=None):
                    calls.append((command,env))
                    target=Path(command[command.index('--prefix')+1]);toolkit(target)
            missing=CudaToolkitUnavailable(dict(detected_toolkits=[],searched_nvcc_paths=[],probe_errors=[]))
            installed=dict(home=str(cache/'cuda_toolkit_12_8_1__private_nvidia_conda_environment'),version='12.8',adapted_from_phase1_cu128=False)
            with patch('monkey_textdet.bootstrap.find_cuda',side_effect=[missing,installed]),\
                 patch('monkey_textdet.bootstrap.find_conda',return_value='/datastore/user/miniconda3/bin/conda'):
                result=ensure_cuda(cache,Process(),dict(auto_install_private_cuda128_if_missing=True),{})
            self.assertEqual(result,installed)
            command,env=calls[0]
            self.assertIn('cuda-toolkit=12.8.1',command)
            self.assertIn('nvidia/label/cuda-12.8.1',command)
            self.assertIn('--override-channels',command)
            self.assertIn('--strict-channel-priority',command)
            self.assertEqual(Path(command[command.index('--prefix')+1]).parent,cache)
            self.assertNotIn('base',command)
            self.assertEqual(env['CONDA_PKGS_DIRS'],str(cache/'conda_packages__private_cuda128'))

    def test_explicit_cuda_failure_does_not_install_fallback(self):
        with tmp() as d:
            class Process:log_dir=Path(d)/'logs'
            missing=CudaToolkitUnavailable(dict(detected_toolkits=[],searched_nvcc_paths=[],probe_errors=[]))
            with patch('monkey_textdet.bootstrap.find_cuda',side_effect=missing),\
                 patch('monkey_textdet.bootstrap.find_conda') as conda:
                with self.assertRaises(CudaToolkitUnavailable):
                    ensure_cuda(Path(d),Process(),dict(cuda_home='/configured/but/missing',auto_install_private_cuda128_if_missing=True),{})
            conda.assert_not_called()

    def test_private_toolkit_completeness_requires_development_files(self):
        with tmp() as d:
            root=Path(d);(root/'bin').mkdir();(root/'bin/nvcc').write_text('fixture')
            self.assertFalse(private_toolkit_complete(root))
            toolkit(root);self.assertTrue(private_toolkit_complete(root))

    def test_existing_compatible_toolkit_does_not_trigger_conda_install(self):
        with tmp() as d:
            class Process:log_dir=Path(d)/'logs'
            existing=dict(home='/site/cuda128',version='12.8',adapted_from_phase1_cu128=False)
            with patch('monkey_textdet.bootstrap.find_cuda',return_value=existing),\
                 patch('monkey_textdet.bootstrap.find_conda') as conda:
                self.assertEqual(ensure_cuda(Path(d),Process(),dict(auto_install_private_cuda128_if_missing=True),{}),existing)
            conda.assert_not_called()

    def test_disabled_auto_provisioning_preserves_missing_toolkit_error(self):
        with tmp() as d:
            class Process:log_dir=Path(d)/'logs'
            missing=CudaToolkitUnavailable(dict(detected_toolkits=[],searched_nvcc_paths=[],probe_errors=[]))
            with patch('monkey_textdet.bootstrap.find_cuda',side_effect=missing),\
                 patch('monkey_textdet.bootstrap.find_conda') as conda:
                with self.assertRaises(CudaToolkitUnavailable):
                    ensure_cuda(Path(d),Process(),dict(auto_install_private_cuda128_if_missing=False),{})
            conda.assert_not_called()


if __name__=='__main__':unittest.main()
