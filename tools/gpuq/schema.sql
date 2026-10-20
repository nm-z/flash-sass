PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS schema_version(version INTEGER PRIMARY KEY);
INSERT OR IGNORE INTO schema_version VALUES(1);
CREATE TABLE IF NOT EXISTS jobs(
 id TEXT PRIMARY KEY, parent_id TEXT, replay_of TEXT, label TEXT,
 host TEXT NOT NULL, boot_id TEXT, pid INTEGER, process_start TEXT,
 cwd TEXT NOT NULL, argv_json TEXT NOT NULL, environment_json TEXT NOT NULL,
 submitted_ns INTEGER NOT NULL, started_ns INTEGER, ended_ns INTEGER,
 status TEXT NOT NULL, exit_code INTEGER, termination_signal INTEGER,
 ticket TEXT, exclusions_json TEXT NOT NULL, profile_version INTEGER NOT NULL DEFAULT 1,
 origin TEXT NOT NULL DEFAULT 'live', source_key TEXT UNIQUE
);
CREATE INDEX IF NOT EXISTS jobs_time ON jobs(submitted_ns);
CREATE TABLE IF NOT EXISTS collectors(
 job_id TEXT NOT NULL REFERENCES jobs(id), name TEXT NOT NULL,
 status TEXT NOT NULL, reason TEXT, config_json TEXT NOT NULL DEFAULT '{}',
 elapsed_ns INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(job_id,name)
);
CREATE TABLE IF NOT EXISTS documents(
 id INTEGER PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
 kind TEXT NOT NULL, source TEXT NOT NULL, ordinal INTEGER NOT NULL,
 observed_ns INTEGER NOT NULL, bytes INTEGER NOT NULL, sha256 TEXT NOT NULL,
 codec TEXT NOT NULL, data BLOB NOT NULL,
 UNIQUE(job_id,kind,source,ordinal)
);
CREATE INDEX IF NOT EXISTS documents_job ON documents(job_id,kind,id);
CREATE VIRTUAL TABLE IF NOT EXISTS document_search USING fts5(text,content='');
CREATE TABLE IF NOT EXISTS metric_definitions(
 name TEXT PRIMARY KEY, unit TEXT NOT NULL, description TEXT NOT NULL,
 definition_version INTEGER NOT NULL DEFAULT 1, collector TEXT NOT NULL DEFAULT 'timing'
);
CREATE TABLE IF NOT EXISTS samples(
 id INTEGER PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
 document_id INTEGER REFERENCES documents(id), metric TEXT NOT NULL REFERENCES metric_definitions(name),
 observed_ns INTEGER, entity TEXT NOT NULL DEFAULT 'job',
 dimensions_json TEXT NOT NULL DEFAULT '{}', value_json TEXT NOT NULL,
 value_number REAL, value_integer INTEGER
);
CREATE INDEX IF NOT EXISTS samples_compare ON samples(metric,job_id,entity);
CREATE TABLE IF NOT EXISTS records(
 id INTEGER PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
 document_id INTEGER NOT NULL REFERENCES documents(id), row_number INTEGER NOT NULL,
 kind TEXT NOT NULL, dimensions_json TEXT NOT NULL, data_json TEXT NOT NULL,
 valid INTEGER NOT NULL DEFAULT 1, UNIQUE(document_id,row_number,kind)
);
CREATE INDEX IF NOT EXISTS records_job ON records(job_id,kind);
CREATE TABLE IF NOT EXISTS file_hashes(
 host TEXT NOT NULL, path TEXT NOT NULL, device INTEGER, inode INTEGER,
 bytes INTEGER, mtime_ns INTEGER, ctime_ns INTEGER, sha256 TEXT NOT NULL,
 PRIMARY KEY(host,path)
);
CREATE VIEW IF NOT EXISTS metric_history AS
 SELECT j.id AS job_id,j.label,j.host,j.submitted_ns,j.status,j.exit_code,j.exclusions_json,
 s.id AS sample_id,s.metric,d.unit,s.observed_ns,s.entity,s.dimensions_json,
 s.value_json,s.value_number,s.value_integer,s.document_id
 FROM samples s JOIN jobs j ON j.id=s.job_id JOIN metric_definitions d ON d.name=s.metric;
CREATE VIEW IF NOT EXISTS phase_intervals AS
 SELECT job_id,document_id,row_number,dimensions_json,data_json,
 CAST(json_extract(data_json,'$.wait_start_ns') AS INTEGER)-CAST(json_extract(data_json,'$.start_ns') AS INTEGER) AS dispatch_ns,
 CAST(json_extract(data_json,'$.wait_end_ns') AS INTEGER)-CAST(json_extract(data_json,'$.wait_start_ns') AS INTEGER) AS exchange_wait_ns,
 CAST(json_extract(data_json,'$.body_end_ns') AS INTEGER)-CAST(json_extract(data_json,'$.wait_end_ns') AS INTEGER) AS body_ns,
 CAST(json_extract(data_json,'$.end_ns') AS INTEGER)-CAST(json_extract(data_json,'$.body_end_ns') AS INTEGER) AS barrier_ns,
 CAST(json_extract(data_json,'$.ple_wait_end_ns') AS INTEGER)-CAST(json_extract(data_json,'$.ple_wait_start_ns') AS INTEGER) AS ple_wait_ns
 FROM records WHERE kind='phase' AND valid=1;
CREATE VIEW IF NOT EXISTS job_metric_status AS
 SELECT j.id AS job_id,j.label,j.status AS job_status,d.name AS metric,d.unit,d.collector,
 CASE WHEN c.status='excluded' THEN 'excluded'
      WHEN EXISTS(SELECT 1 FROM samples s WHERE s.job_id=j.id AND s.metric=d.name) THEN 'recorded'
      WHEN d.collector='trace' AND EXISTS(SELECT 1 FROM records r WHERE r.job_id=j.id AND r.kind='phase' AND r.valid=1) THEN 'recorded'
      WHEN c.status IN ('unavailable','historical_unknown','partial') THEN c.status
      ELSE 'not_recorded' END AS measurement_status,
 c.reason
 FROM jobs j CROSS JOIN metric_definitions d LEFT JOIN collectors c ON c.job_id=j.id AND c.name=d.collector;
