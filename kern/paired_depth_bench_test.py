#!/usr/bin/env python3
import copy,unittest
from paired_depth_bench import phase_summary,host_summary,matched
META=dict(first="2048",count="2",nph="2",ctas="1")
def fixture():
    rows=[]
    for i in range(4):
        start=100+i*50+(10 if i>=2 else 0)
        z=dict(position=str(2048+i//2),phase=str(i%2),opcode="PLEROWS" if i%2 else "IDXSEL",selected_keys=str(2049+i//2))
        z.update(zip(("start_ns","wait_start_ns","wait_end_ns","body_end_ns","end_ns"),map(str,range(start,start+41,10))))
        z.update(ple_wait_start_ns="0",ple_wait_end_ns="0",publication_ns=str(start+50) if i%2 else "0")
        if i==3:z.update(ple_wait_start_ns=str(start+22),ple_wait_end_ns=str(start+26))
        rows.append(z)
    return rows
class Tests(unittest.TestCase):
    def test_partition_and_phase_zero(self):
        r=phase_summary(fixture(),META)
        self.assertEqual(r["intervals_ns"],[110])
        self.assertEqual(r["tokens"][1]["phase_ns"],80)
        self.assertEqual(r["tokens"][1]["between_phase_ns"],10)
        self.assertEqual(r["tokens"][1]["publication_footer_ns"],10)
        self.assertEqual(r["op_counts"],{"IDXSEL":1,"PLEROWS":1})
        self.assertEqual(r["op_mean_us"]["PLEROWS"]["ple_wait"],.004)
    def test_64_bit_epoch(self):
        rows=fixture()
        for r in rows:
            for k in list(r):
                if k.endswith("_ns") and int(r[k]):r[k]=str(int(r[k])+(1<<32))
        self.assertEqual(phase_summary(rows,META)["intervals_ns"],[110])
    def test_missing(self):
        with self.assertRaises(ValueError):phase_summary(fixture()[:-1],META)
    def test_duplicate(self):
        r=fixture();r[1]=r[0]
        with self.assertRaises(ValueError):phase_summary(r,META)
    def test_missing_publication(self):
        r=fixture();r[-1]["publication_ns"]="0"
        with self.assertRaises(ValueError):phase_summary(r,META)
    def test_wrong_layout(self):
        with self.assertRaises(ValueError):phase_summary(fixture(),dict(META,ctas="16"))
    def test_invalid_ple_wait(self):
        r=fixture();r[-1]["ple_wait_end_ns"]=r[-1]["publication_ns"]
        with self.assertRaises(ValueError):phase_summary(r,META)
    def test_single_publication(self):
        r=phase_summary(fixture()[:2],dict(META,count="1"))
        self.assertEqual(r["intervals_ns"],[])
        self.assertEqual(r["decode_ms"],0)
    def test_manifest_gate(self):
        matched(dict(kv_host="0",layers="48/48"),dict(kv_host="1",layers="48/48"),{"kv_host"})
        with self.assertRaises(ValueError):matched(dict(kv_host="0",layers="12/48"),dict(kv_host="1",layers="48/48"),{"kv_host"})
    def test_host_short_read(self):
        r=dict(position="2048",observed_ns="100",callback_start_ns="101",callback_end_ns="102",ple_staging_start_ns="1",ple_staging_end_ns="2",ple_ready_ns="3",ple_short_reads="1")
        with self.assertRaises(ValueError):host_summary([r],dict(first="2048",count="1",clock="CLOCK_MONOTONIC"))
if __name__=="__main__":unittest.main()
