"""FP32 prefix bounds retain exact SSD results and traversal, including full D."""
import argparse,os,subprocess,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from threadpoolctl import threadpool_limits
from test_mixed_pivot_pipeline import builder,bin_write,make_sectors
from test_mixed_subpq_pipeline import write_subpq
def main():
    p=argparse.ArgumentParser();p.add_argument('--search-binary',required=True);a=p.parse_args()
    env={k:v for k,v in os.environ.items() if not k.startswith('SKIPDISK_')}
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SKIPDISK_BENCH_L_VALUES='30,320',SKIPDISK_BENCH_REPEATS='1')
    with tempfile.TemporaryDirectory(prefix='fp32-filter-') as tmp:
        root=Path(tmp);rng=np.random.default_rng(471)
        base=rng.normal(size=(320,13)).astype('f4');q=rng.normal(size=(12,13)).astype('f4')
        bin_write(root/'base.bin',base);bin_write(root/'query.bin',q);make_sectors(root/'graph.index',base,True)
        prefix=str(root/'mixed');pq=str(root/'pq')
        with threadpool_limits(limits=1):
            builder.build(SimpleNamespace(pca_base=str(root/'base.bin'),graph_index=str(root/'graph.index'),index_prefix=prefix,
                m=2,n=3,train_size=320,iterations=3,batch_size=73,threads=1,seed=42))
        make_sectors(Path(prefix+'_data_sector.bin'),base,False);write_subpq(pq,base,7,3)
        cmd=[a.search_binary,prefix,str(root/'query.bin'),'-','10','30','4','1','-',str(root/'baseline'),pq,'--sweep']
        subprocess.run(cmd+['--no-bound'],env=env,check=True,capture_output=True)
        for d in (1,7,13):
            test=cmd.copy();test[9]=str(root/f'd{d}')
            run=subprocess.run(test+['--fp32-filter',str(root/'base.bin'),str(d)],env=env,text=True,capture_output=True)
            assert run.returncode==0,run.stdout+run.stderr
            assert f'FP32FilterDim={d}' in run.stdout+run.stderr
            for length in (30,320):
                for kind in ('ids','dists','traversal'):
                    assert Path(test[9]+f'_L{length}_B1_R1_{kind}.bin').read_bytes()==Path(cmd[9]+f'_L{length}_B0_R1_{kind}.bin').read_bytes()
        for d in ('0','14','-1'):
            run=subprocess.run(cmd+['--fp32-filter',str(root/'base.bin'),d],env=env,capture_output=True)
            assert run.returncode!=0
    print('FP32_FILTER_REGRESSION_PASSED')
if __name__=='__main__':main()
