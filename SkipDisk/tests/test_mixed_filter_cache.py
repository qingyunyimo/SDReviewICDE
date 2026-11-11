"""Partial mixed pivots: actual sparse storage, exact bounds and estimation."""
import argparse,os,re,struct,subprocess,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from threadpoolctl import threadpool_limits
from test_mixed_pivot_pipeline import builder,bin_write,make_sectors
from test_mixed_subpq_pipeline import write_subpq

def main():
    p=argparse.ArgumentParser();p.add_argument('--search-binary',required=True);a=p.parse_args()
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for key in list(env):
        if key.startswith('SKIPDISK_'):env.pop(key)
    with tempfile.TemporaryDirectory(prefix='mixed-cache-') as tmp:
        w=Path(tmp);rng=np.random.default_rng(16);n=320
        base=(rng.normal(size=(n,29))*np.geomspace(3,.1,29)).astype('<f4')
        query=rng.normal(size=(12,29)).astype('<f4')
        bin_write(w/'base.bin',base);bin_write(w/'query.bin',query)
        make_sectors(w/'graph.index',base,True)
        pq=str(w/'pq');write_subpq(pq,base,7,3)
        def vector(path,x,dtype):
            x=np.asarray(x,dtype=dtype);path.write_bytes(struct.pack('<I',len(x))+x.tobytes());return str(path)
        eps=vector(w/'eps',np.zeros(30),'<f4');lmd=vector(w/'lmd',[1]*21+[100]*8,'<f4')
        unity=vector(w/'unity',[1]*21+[0]*8,'<f4')
        rank=rng.permutation(n);rankfile=vector(w/'rank.id',rank,'<u4')
        def values(log):return {k:float(v) for k,v in re.findall(r'([A-Za-z0-9@]+)=([0-9][0-9.eE+-]*)',log)}
        for mode,m,nd,p2,p4 in [('mixed',1,2,2,8),('bf16',21,0,0,0)]:
            prefix=str(w/mode)
            with threadpool_limits(limits=1):
                builder.build(SimpleNamespace(pca_base=str(w/'base.bin'),graph_index=str(w/'graph.index'),index_prefix=prefix,
                    m=m,n=nd,pq2_dims=p2,pq4_dims=p4,pivot_dim=21,train_size=n,iterations=3,batch_size=73,threads=1,seed=42))
            make_sectors(Path(prefix+'_data_sector.bin'),base,False)
            vector(Path(prefix+'_degree_sorted.id'),rank,'<u4')
            serial=0
            def run(count=None,extra=(),disk=False,threads=1,beam=4,ok=True,ids=rankfile,nav=True):
                nonlocal serial
                serial+=1;out=str(w/f'out{mode}{serial}')
                cmd=[a.search_binary,prefix,str(w/'query.bin'),'-','10','320',str(beam),str(threads),'-',out]
                if nav:cmd.append(pq)
                if not disk:cmd+=['--memory-base',str(w/'base.bin')]
                if count is not None:cmd+=['--filter-cache',ids,str(count)]
                r=subprocess.run(cmd+list(extra),env=env,text=True,capture_output=True,timeout=120)
                assert (r.returncode==0)==ok,r.stdout+r.stderr
                return out,r.stdout+r.stderr
            ref,_=run(extra=['--no-bound'])
            def identical(out,kinds=('ids','dists','traversal')):
                for kind in kinds:assert Path(out+f'_{kind}.bin').read_bytes()==Path(ref+f'_{kind}.bin').read_bytes()
            for count in (0,1,80,n):
                for disk,threads,beam in ((False,1,1),(True,1,1),(True,2,4),(True,2,32)):
                    out,log=run(count,disk=disk,threads=threads,beam=beam);identical(out)
                    v=values(log)
                    assert v['MixedCachePoints']==count and v['MixedCacheMapBytes']==4*n
                    # Code v3 padded payload: mixed=8+4 bytes, BF16=44+4 bytes.
                    assert v['MixedCachePayloadBytes']==count*(12 if mode=='mixed' else 48)
                    assert v['MeanBounds']==count and v['MeanCandidates']==n and v['ShortResults']==0
                    assert v['MeanExact']>=n-count
                    if count==0:assert v['MeanExact']==n
                out,_=run(count,extra=['--estimation',unity,eps]);identical(out)
                out,log=run(count,extra=['--estimation',lmd,eps],disk=True,threads=2)
                v=values(log);assert v['MeanExact']>=n-count and v['MeanBounds']==count and v['ShortResults']==0
                identical(out,('traversal',))
                ids=np.fromfile(out+'_ids.bin','<u4',offset=8).reshape(12,10)
                assert np.all(ids<n) and all(len(set(row))==10 for row in ids)
                actual=builder.read_bin(out+'_dists.bin')
                exact=((query[:,None].astype('f8')-base[ids].astype('f8'))**2).sum(axis=2)
                assert np.allclose(actual,exact,rtol=1e-5,atol=1e-5)
                if count==0:identical(out)
            out,_=run(80,ids='-');identical(out)
            plain,_=run(extra=['--estimation',lmd,eps])
            cached,_=run(n,extra=['--estimation',lmd,eps])
            for kind in ('ids','dists','traversal'):
                assert Path(plain+f'_{kind}.bin').read_bytes()==Path(cached+f'_{kind}.bin').read_bytes()
            out,log=run(80,extra=['--no-bound']);identical(out);assert values(log)['MeanBounds']==0
            out,_=run(80,extra=['--quantized-only'],disk=True)
            full,_=run(extra=['--quantized-only'],disk=True)
            assert Path(out+'_ids.bin').read_bytes()==Path(full+'_ids.bin').read_bytes()
            for count in (-1,n+1,'x'):run(count,ok=False)
            run(80,nav=False,ok=False)
            run(2,ids=vector(w/'duplicate',[0,0],'<u4'),ok=False)
            run(1,ids=vector(w/'range',[n],'<u4'),ok=False)
            run(80,ids=vector(w/'short',[1],'<u4'),ok=False)
            run(1,ids=str(w/'missing'),ok=False)
            malformed=w/'malformed';malformed.write_bytes(struct.pack('<I',2)+struct.pack('<I',0))
            run(1,ids=str(malformed),ok=False)
        print('MIXED_FILTER_CACHE_TEST_PASSED')

if __name__=='__main__':main()
