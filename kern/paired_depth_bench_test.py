#!/usr/bin/env python3
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from paired_depth_bench import phase_summary,host_summary,matched,driver_snapshot,core_records,trace_lifecycle
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
    def test_failed_driver_query_cannot_pass(self):
        with patch("paired_depth_bench.run",return_value=dict(exit=2,stdout="Failed to parse flags",stderr="")):
            with self.assertRaises(ValueError):driver_snapshot()
        with patch("paired_depth_bench.run",return_value=dict(exit=0,stdout="<nvidia_smi_log></nvidia_smi_log>",stderr="")):
            self.assertEqual(driver_snapshot()["exit"],0)
    def test_core_launch_records(self):
        record=dict(argv=["flashcore","-","libengine.so","manifest"],launch_environment=dict(FLASH_MAXT="8192",CUDA_VISIBLE_DEVICES="0,1",CUDA_DEVICE_ORDER="PCI_BUS_ID"),engine_reload_commands=[],limitations=[])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"launch.json";p.write_text(json.dumps(dict(host=record,vram=record)))
            self.assertEqual(core_records(p)["host"]["argv"][0],"flashcore")
            bad=copy.deepcopy(record);del bad["launch_environment"]["FLASH_MAXT"]
            p.write_text(json.dumps(dict(host=bad,vram=record)))
            with self.assertRaises(ValueError):core_records(p)
    def test_host_short_read(self):
        r=dict(position="2048",observed_ns="100",callback_start_ns="101",callback_end_ns="102",ple_staging_start_ns="1",ple_staging_end_ns="2",ple_ready_ns="3",ple_short_reads="1")
        with self.assertRaises(ValueError):host_summary([r],dict(first="2048",count="1",clock="CLOCK_MONOTONIC"))
    def test_legacy_lifecycle_is_analysis_only(self):
        result=phase_summary(fixture(),META)
        self.assertEqual(result["lifecycle"],dict(reported=False,status="legacy_unreported"))
        with self.assertRaises(ValueError):trace_lifecycle(META,require_current=True)
    def test_current_lifecycle_success(self):
        meta=dict(META,status="completed",aborted="0",ended="0",completed="2050",requested="2050")
        result=trace_lifecycle(meta,require_current=True,expected_requested=2050,allow_ended=False)
        self.assertTrue(result["reported"])
        self.assertEqual(phase_summary(fixture(),meta)["lifecycle"]["completed"],2050)
    def test_failed_lifecycle_cannot_be_historical_success(self):
        for status in ("aborted","argument_error","image_error","device_error","running","none"):
            meta=dict(META,status=status,aborted="1" if status=="aborted" else "0",ended="1",completed="2050",requested="2050")
            with self.subTest(status=status):
                with self.assertRaises(ValueError):phase_summary(fixture(),meta)
                rows=[dict(position=str(2048+i),observed_ns=str(100+10*i),callback_start_ns=str(101+10*i),callback_end_ns=str(102+10*i),ple_staging_start_ns="0",ple_staging_end_ns="0",ple_ready_ns="0",ple_short_reads="0") for i in range(2)]
                with self.assertRaises(ValueError):host_summary(rows,dict(meta,clock="CLOCK_MONOTONIC"))
    def test_stop_is_analyzable_but_not_fixed_continuation(self):
        meta=dict(META,count="1",status="completed",aborted="0",ended="1",completed="2049",requested="2050")
        self.assertEqual(phase_summary(fixture()[:2],meta)["lifecycle"]["ended"],1)
        with self.assertRaises(ValueError):trace_lifecycle(meta,require_current=True,allow_ended=False)
    def test_partial_or_malformed_lifecycle(self):
        with self.assertRaises(ValueError):trace_lifecycle(dict(META,status="completed"))
        good=dict(META,status="completed",aborted="0",ended="0",completed="2050",requested="2050")
        for change in (dict(aborted="2"),dict(ended="false"),dict(completed="-1"),dict(requested="nan"),dict(completed="2049"),dict(completed="2051")):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):trace_lifecycle(dict(good,**change))
    def test_wrong_request_or_excess_window_rejected(self):
        meta=dict(META,status="completed",aborted="0",ended="0",completed="2050",requested="2050")
        with self.assertRaises(ValueError):trace_lifecycle(meta,require_current=True,expected_requested=2051)
        with self.assertRaises(ValueError):trace_lifecycle(dict(meta,count="3"))
if __name__=="__main__":unittest.main()
