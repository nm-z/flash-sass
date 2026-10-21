-- Open the canonical database and ATTACH one original snapshot AS incoming.
-- Every mismatch result must be zero. Document IDs are intentionally remapped.
CREATE TEMP VIEW imported_doc_map AS
 SELECT o.id AS old_id,n.id AS new_id FROM incoming.documents o
 JOIN main.documents n ON n.job_id=o.job_id AND n.kind=o.kind
 AND n.source=o.source AND n.ordinal=o.ordinal;
SELECT 'missing_or_changed_jobs',count(*) FROM
 (SELECT * FROM incoming.jobs EXCEPT SELECT * FROM main.jobs);
SELECT 'missing_or_changed_collectors',count(*) FROM
 (SELECT * FROM incoming.collectors EXCEPT SELECT * FROM main.collectors);
SELECT 'missing_or_changed_documents',count(*) FROM incoming.documents o
 LEFT JOIN main.documents n ON n.job_id=o.job_id AND n.kind=o.kind
 AND n.source=o.source AND n.ordinal=o.ordinal
 WHERE n.id IS NULL OR n.observed_ns!=o.observed_ns OR n.bytes!=o.bytes
 OR n.sha256!=o.sha256 OR n.codec!=o.codec OR n.data!=o.data;
SELECT 'sample_count_difference',
 (SELECT count(*) FROM main.samples WHERE job_id IN (SELECT id FROM incoming.jobs))
 -(SELECT count(*) FROM incoming.samples);
SELECT 'missing_or_changed_samples',count(*) FROM (
 SELECT s.job_id,m.new_id AS document_id,s.metric,s.observed_ns,s.entity,
 s.dimensions_json,s.value_json,s.value_number,s.value_integer
 FROM incoming.samples s LEFT JOIN imported_doc_map m ON m.old_id=s.document_id
 EXCEPT SELECT job_id,document_id,metric,observed_ns,entity,dimensions_json,
 value_json,value_number,value_integer FROM main.samples
 WHERE job_id IN (SELECT id FROM incoming.jobs));
SELECT 'record_count_difference',
 (SELECT count(*) FROM main.records WHERE job_id IN (SELECT id FROM incoming.jobs))
 -(SELECT count(*) FROM incoming.records);
SELECT 'missing_or_changed_records',count(*) FROM (
 SELECT r.job_id,m.new_id AS document_id,r.row_number,r.kind,r.dimensions_json,
 r.data_json,r.valid FROM incoming.records r JOIN imported_doc_map m ON m.old_id=r.document_id
 EXCEPT SELECT job_id,document_id,row_number,kind,dimensions_json,data_json,valid
 FROM main.records WHERE job_id IN (SELECT id FROM incoming.jobs));
SELECT 'foreign_key_errors',count(*) FROM pragma_foreign_key_check;
