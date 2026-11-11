"""Mixed estimation: all precision tiers, prefix semantics and RAM/SSD paths."""
import argparse, os, re, struct, subprocess, tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from threadpoolctl import threadpool_limits
from test_mixed_pivot_pipeline import builder, bin_write, make_sectors
from test_mixed_subpq_pipeline import write_subpq

def main():
    p=argparse.ArgumentParser();p.add_argument('--search-binary',required=True);args=p.parse_args()
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for key in list(env):
        if key.startswith('SKIPDISK_BENCH_'):env.pop(key)
    with tempfile.TemporaryDirectory(prefix='mixed-estimation-') as tmp:
        w=Path(tmp);rng=np.random.default_rng(61)
        base=(rng.normal(size=(320,29))*np.geomspace(3,.1,29)).astype('<f4')
        queries=rng.normal(size=(12,29)).astype('<f4')
        bin_write(w/'base.bin',base);bin_write(w/'query.bin',queries)
        make_sectors(w/'graph.index',base,True)
        pq=str(w/'pq');write_subpq(pq,base,7,3)
        def vec(name,values):
            a=np.asarray(values,dtype='<f4');path=w/name
            path.write_bytes(struct.pack('<I',len(a))+a.tobytes());return str(path)
        eps=vec('eps',np.zeros(30))
        for d in (21,29):
            prefix=str(w/f'mix{d}')
            with threadpool_limits(limits=1):
                builder.build(SimpleNamespace(pca_base=str(w/'base.bin'),graph_index=str(w/'graph.index'),
                    index_prefix=prefix,m=1,n=2,pq2_dims=2,pq4_dims=8,pivot_dim=d,
                    train_size=320,iterations=3,batch_size=73,threads=1,seed=42))
            make_sectors(Path(prefix+'_data_sector.bin'),base,False)
            lmd=vec('lmd',[1]*d+[0]*(29-d))
            aggressive=vec('aggressive',[1]*d+[100]*(29-d))
            def run(name,extra=(),disk=False,nav=True,threads=1,ok=True):
                out=str(w/name)
                cmd=[args.search_binary,prefix,str(w/'query.bin'),'-','10','320','4',str(threads),'-',out]
                if nav:cmd.append(pq)
                if not disk:cmd+=['--memory-base',str(w/'base.bin')]
                r=subprocess.run(cmd+list(extra),env=env,text=True,capture_output=True,timeout=120)
                assert (r.returncode==0)==ok,r.stdout+r.stderr
                return out,r.stdout
            ref,ref_log=run(f'ref{d}')
            for disk in (False,True):
                for threads in (1,2):
                    out,log=run(f'unity{d}{disk}{threads}',['--estimation',lmd,eps],disk,threads=threads)
                    assert 'FilteringMode=MixedEstimation EstimationRatio=1 ' in log
                    for kind in ('ids','dists','traversal'):
                        assert Path(out+f'_{kind}.bin').read_bytes()==Path(ref+f'_{kind}.bin').read_bytes()
                    out,log=run(f'est{d}{disk}{threads}',['--estimation',aggressive,eps],disk,threads=threads)
                    assert f'FilteringDim={d}' in log and 'ShortResults=0' in log
                    assert Path(out+'_traversal.bin').read_bytes()==Path(ref+'_traversal.bin').read_bytes()
                    ids=np.fromfile(out+'_ids.bin','<u4',offset=8).reshape(12,10)
                    assert np.all(ids<320) and all(len(set(row))==10 for row in ids)
                    distances=builder.read_bin(out+'_dists.bin')
                    expected=((queries[:,None].astype('f8')-base[ids].astype('f8'))**2).sum(axis=2)
                    assert np.allclose(distances,expected,rtol=1e-5,atol=1e-5)
                    if d==21:
                        assert 'EstimationRatio=0.0255786' in log
                        if not disk:
                            exact=lambda s:float(re.search(r'MeanExact=([\d.e+-]+)',s)[1])
                            assert exact(log)<exact(ref_log)
            # Mixed navigation, not only independent subPQ.
            a,_=run('plain',nav=False)
            b,_=run('plainest',['--estimation',lmd,eps],nav=False)
            assert Path(a+'_ids.bin').read_bytes()==Path(b+'_ids.bin').read_bytes()
            for flag in ('--no-bound','--compare-bounds','--quantized-only'):
                run('invalid',['--estimation',lmd,eps,flag],ok=False)
            for bad in (vec('zero',np.zeros(29)),vec('nan',[float('nan')]*29),
                        vec('negative',[-1]*29),vec('wrong',np.ones(28))):
                run('invalid',['--estimation',bad,eps],ok=False)
            if d<29:
                # Generator column d-1 is the d-coordinate squared-distance error.
                # Set adjacent entry differently to catch an off-by-one regression.
                errors=np.zeros(30);errors[d-1]=1;errors[d]=3
                _,log=run('epsilon',['--estimation',aggressive,vec('eps_nonzero',errors)])
                ratio=float(re.search(r'EstimationRatio=([\d.e+-]+)',log)[1])
                assert np.isclose(ratio,2*d/(d+100*(29-d)),rtol=1e-5)
                out,log=run('loose',['--estimation',lmd,vec('eps_loose',errors)])
                assert 'EstimationRatio=2 ' in log
                assert Path(out+'_ids.bin').read_bytes()==Path(ref+'_ids.bin').read_bytes()
                _,log=run('exact_length',['--estimation',lmd,vec('exact_eps',errors[:d])])
                assert 'EstimationRatio=2 ' in log
                run('invalid',['--estimation',lmd,vec('short',np.zeros(d-1))],ok=False)
                # Unused negative quantiles must not reject or change this search.
                errors=np.zeros(30);errors[:3]=[-.320677,-.075485,-.015555]
                out,log=run('unused_negative',['--estimation',lmd,vec('unused_eps',errors)])
                assert 'EstimationRatio=1 ' in log
                for kind in ('ids','dists','traversal'):
                    assert Path(out+f'_{kind}.bin').read_bytes()==Path(ref+f'_{kind}.bin').read_bytes()
                # Selected negative epsilon retains its value (not clamped to zero).
                errors[d-1]=-.5
                for disk in (False,True):
                    out,log=run('negative_selected',['--estimation',lmd,vec('negative_eps',errors)],disk=disk)
                    assert 'EstimationRatio=0.5 ' in log and 'ShortResults=0' in log
                    ids=np.fromfile(out+'_ids.bin','<u4',offset=8).reshape(12,10)
                    expected=((queries[:,None].astype('f8')-base[ids].astype('f8'))**2).sum(axis=2)
                    assert np.allclose(builder.read_bin(out+'_dists.bin'),expected,rtol=1e-5,atol=1e-5)
                for invalid in (-1.,-1.01,float('nan'),float('inf')):
                    errors[d-1]=invalid
                    run('invalid',['--estimation',lmd,vec('invalid_eps',errors)],ok=False)
        print('MIXED_ESTIMATION_TEST_PASSED')

if __name__=='__main__':main()
