"""Separate navigation coordinates from filtering/rerank; SSD correctness."""
import argparse,os,subprocess,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from threadpoolctl import threadpool_limits
from test_mixed_pivot_pipeline import builder,bin_write,make_sectors
from test_mixed_subpq_pipeline import write_subpq

def main():
    p=argparse.ArgumentParser();p.add_argument('--search-binary',required=True);args=p.parse_args()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SKIPDISK_BENCH_REPEATS='1',SKIPDISK_BENCH_L_VALUES='30,320')
    for key in list(env):
        if key.startswith('SKIPDISK_') and key not in ('SKIPDISK_BENCH_REPEATS','SKIPDISK_BENCH_L_VALUES'):env.pop(key)
    with tempfile.TemporaryDirectory(prefix='independent-navigation-') as t:
        r=Path(t);rng=np.random.default_rng(74);x=rng.normal(size=(320,12)).astype('<f4');q=rng.normal(size=(12,12)).astype('<f4')
        perm=rng.permutation(12);nx=x[:,perm].copy();nq=q[:,perm].copy()
        bin_write(r/'navq.bin',nq);write_subpq(str(r/'pq'),nx,8,2)
        for name,base,query in [('raw',x,q),('nav',nx,nq)]:
            bin_write(r/(name+'.bin'),base);bin_write(r/(name+'q.bin'),query)
            make_sectors(r/(name+'graph.index'),base,True)
            with threadpool_limits(limits=1):
                builder.build(SimpleNamespace(pca_base=str(r/(name+'.bin')),graph_index=str(r/(name+'graph.index')),index_prefix=str(r/name),m=0,n=12,train_size=320,iterations=3,batch_size=73,threads=1,seed=42))
            make_sectors(r/(name+'_data_sector.bin'),base,False)
        for name in ('raw','nav'):
            cmd=[args.search_binary,str(r/name),str(r/(name+'q.bin')),'-','10','30','4','1','-',str(r/(name+'out')),str(r/'pq'),'--compare-bounds']
            if name=='raw':cmd+=['--navigation-query',str(r/'navq.bin')]
            result=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=120)
            assert result.returncode==0,result.stdout+result.stderr
            assert result.stdout.count('ResultsAndTraversalVerified=1')==4
        for length in (30,320):
            for mode in (0,1):
                for kind in ('ids','traversal'):
                    suffix=f'_L{length}_B{mode}_R1_{kind}.bin'
                    assert (r/('rawout'+suffix)).read_bytes()==(r/('navout'+suffix)).read_bytes()
        ids=np.fromfile(r/'rawout_L320_B1_R1_ids.bin','<u4',offset=8).reshape(12,10)
        exact=((q[:,None].astype('f8')-x.astype('f8'))**2).sum(2)
        assert np.array_equal(ids,np.argsort(exact,axis=1)[:,:10])
        bad=r/'bad.bin';bin_write(bad,nq[:3])
        cmd=[args.search_binary,str(r/'raw'),str(r/'rawq.bin'),'-','10','30','4','1','-',str(r/'badout'),str(r/'pq'),'--navigation-query',str(bad)]
        assert subprocess.run(cmd,env=env,capture_output=True,timeout=60).returncode!=0
        invalid=nq.copy();invalid[0,0]=np.nan;bin_write(bad,invalid)
        assert subprocess.run(cmd,env=env,capture_output=True,timeout=60).returncode!=0
        bin_write(bad,nq)
        assert subprocess.run(cmd+['--memory-base',str(r/'raw.bin')],env=env,capture_output=True,timeout=60).returncode!=0
    print('INDEPENDENT_NAVIGATION_QUERY_PASSED')
if __name__=='__main__':main()
