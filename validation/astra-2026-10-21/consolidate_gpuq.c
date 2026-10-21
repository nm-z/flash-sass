/* Recover isolated histories into the canonical GPUQ v2 schema. Each job is
 * atomic. UUIDs and document bytes stay unchanged; integer references are mapped.
 * Existing UUIDs are refused, rather than silently dropping or overwriting data.
 * Compile: cc -O2 -Wall -Wextra -Werror consolidate_gpuq.c -lsqlite3 -lz -lcrypto -o consolidate_gpuq
 * Usage: consolidate_gpuq DESTINATION SOURCE SOURCE_SHA256
 * Back up DESTINATION first. SOURCE must contain terminal jobs only.
 */
#define _POSIX_C_SOURCE 200809L
#include <sqlite3.h>
#include <openssl/sha.h>
#include <zlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static sqlite3 *db;
static void fail(const char *s) { fprintf(stderr,"consolidate_gpuq: %s: %s\n",s,db?sqlite3_errmsg(db):"");if(db)sqlite3_exec(db,"ROLLBACK",NULL,NULL,NULL);exit(1); }
static sqlite3_stmt *prepare(const char *sql) {
    sqlite3_stmt *q=NULL;
    if(sqlite3_prepare_v2(db,sql,-1,&q,NULL)!=SQLITE_OK)fail(sql);
    return q;
}
static void execute(const char *sql) { if(sqlite3_exec(db,sql,NULL,NULL,NULL)!=SQLITE_OK)fail(sql); }
static void bind_text(sqlite3_stmt *q,int index,const char *text) { if(sqlite3_bind_text(q,index,text,-1,SQLITE_TRANSIENT)!=SQLITE_OK)fail("bind"); }
static void job_sql(const char *sql,const char *id) {
    sqlite3_stmt *q=prepare(sql);bind_text(q,1,id);
    if(sqlite3_step(q)!=SQLITE_DONE)fail(sql);
    sqlite3_finalize(q);
}
static sqlite3_int64 number(const char *sql) {
    sqlite3_stmt *q=prepare(sql);
    if(sqlite3_step(q)!=SQLITE_ROW)fail(sql);
    sqlite3_int64 n=sqlite3_column_int64(q,0);sqlite3_finalize(q);return n;
}
static void index_document(sqlite3_int64 id,const void *text,size_t n) {
    sqlite3_stmt *q=prepare("INSERT INTO document_search(rowid,text) VALUES(?,?)");
    sqlite3_bind_int64(q,1,id);sqlite3_bind_text64(q,2,text,n,SQLITE_TRANSIENT,SQLITE_UTF8);
    if(sqlite3_step(q)!=SQLITE_DONE)fail("index document");
    sqlite3_finalize(q);
}
static void digest(const void *data,size_t n,char hex[65]) {
    unsigned char sha[SHA256_DIGEST_LENGTH];
    if(!SHA256(data,n,sha))fail("sha256");
    for(unsigned i=0;i<sizeof sha;i++)sprintf(hex+2*i,"%02x",sha[i]);
}
static void index_copied(void) {
    sqlite3_stmt *q=prepare("SELECT d.id,d.codec,d.bytes,d.data,d.sha256 FROM documents d JOIN doc_map m ON m.new_id=d.id ORDER BY d.id");
    int rc;
    while((rc=sqlite3_step(q))==SQLITE_ROW) {
        sqlite3_int64 length=sqlite3_column_int64(q,2);
        if(length<0||length>512*1024*1024)fail("document size");
        size_t n=(size_t)length;unsigned char *raw=malloc(n+1);
        if(!raw)fail("allocation");
        const char *codec=(const char *)sqlite3_column_text(q,1);
        const void *data=sqlite3_column_blob(q,3);int stored=sqlite3_column_bytes(q,3);
        if(!strcmp(codec,"raw")) {
            if((size_t)stored!=n)fail("raw size");
            if(n)memcpy(raw,data,n);
        } else if(!strcmp(codec,"zlib")) {
            uLongf out=(uLongf)n+1;
            if(uncompress(raw,&out,data,(uLong)stored)!=Z_OK||out!=n)fail("decompression");
        } else fail("codec");
        char hex[65];digest(raw,n,hex);
        if(strcmp(hex,(const char *)sqlite3_column_text(q,4)))fail("document SHA256 mismatch");
        index_document(sqlite3_column_int64(q,0),raw,n);free(raw);
    }
    if(rc!=SQLITE_DONE)fail("read copied documents");
    sqlite3_finalize(q);
}
static void provenance(const char *job,const char *path,const char *hash) {
    sqlite3_stmt *q=prepare("SELECT json_object('source_database',?1,'source_sha256',?2,'document_id_map',json((SELECT json_group_array(json_object('old',old_id,'new',new_id)) FROM doc_map)),'source_file_hash_cache',json((SELECT json_group_array(json_object('host',host,'path',path,'device',device,'inode',inode,'bytes',bytes,'mtime_ns',mtime_ns,'ctime_ns',ctime_ns,'sha256',sha256)) FROM incoming.file_hashes)))");
    bind_text(q,1,path);bind_text(q,2,hash);
    if(sqlite3_step(q)!=SQLITE_ROW)fail("provenance");
    const char *text=(const char *)sqlite3_column_text(q,0);size_t n=(size_t)sqlite3_column_bytes(q,0);char sha[65];digest(text,n,sha);
    sqlite3_stmt *put=prepare("INSERT INTO documents(job_id,kind,source,ordinal,observed_ns,bytes,sha256,codec,data) VALUES(?1,'provenance','gpuq-canonical-consolidation',0,CAST(strftime('%s','now') AS INTEGER)*1000000000,?2,?3,'raw',?4)");
    bind_text(put,1,job);sqlite3_bind_int64(put,2,(sqlite3_int64)n);bind_text(put,3,sha);sqlite3_bind_blob64(put,4,text,n,SQLITE_TRANSIENT);
    if(sqlite3_step(put)!=SQLITE_DONE)fail("store provenance");
    sqlite3_int64 id=sqlite3_last_insert_rowid(db);sqlite3_finalize(put);index_document(id,text,n);sqlite3_finalize(q);
}
int main(int argc,char **argv) {
    if(argc!=4||strlen(argv[3])!=64)fail("usage: consolidate_gpuq DESTINATION SOURCE SOURCE_SHA256");
    if(sqlite3_open_v2(argv[1],&db,SQLITE_OPEN_READWRITE,NULL)!=SQLITE_OK)fail("open destination");
    sqlite3_busy_timeout(db,15000);execute("PRAGMA foreign_keys=ON");
    sqlite3_stmt *q=prepare("ATTACH DATABASE ?1 AS incoming");bind_text(q,1,argv[2]);
    if(sqlite3_step(q)!=SQLITE_DONE)fail("attach source");
    sqlite3_finalize(q);
    if(number("SELECT max(version) FROM schema_version")!=2||number("SELECT max(version) FROM incoming.schema_version")!=2)fail("requires v2 schemas");
    if(number("SELECT count(*) FROM incoming.jobs WHERE status IN ('queued','running','collecting')"))fail("source has live jobs");
    if(number("SELECT count(*) FROM jobs j JOIN incoming.jobs i ON i.id=j.id"))fail("overlapping job UUIDs; no import performed");
    if(number("SELECT count(*) FROM metric_definitions d JOIN incoming.metric_definitions i USING(name) WHERE d.unit!=i.unit"))fail("metric unit conflict");
    char **ids=NULL;size_t count=0;
    q=prepare("SELECT id FROM incoming.jobs ORDER BY submitted_ns");int rc;
    while((rc=sqlite3_step(q))==SQLITE_ROW) {
        char **next=realloc(ids,(count+1)*sizeof *ids);if(!next)fail("job list");ids=next;
        ids[count]=strdup((const char *)sqlite3_column_text(q,0));if(!ids[count])fail("job id");count++;
    }
    if(rc!=SQLITE_DONE)fail("read job list");
    sqlite3_finalize(q);
    execute("CREATE TEMP TABLE doc_map(old_id INTEGER PRIMARY KEY,new_id INTEGER UNIQUE)");
    execute("BEGIN IMMEDIATE; INSERT OR IGNORE INTO metric_definitions SELECT * FROM incoming.metric_definitions; INSERT OR IGNORE INTO file_hashes SELECT * FROM incoming.file_hashes; COMMIT");
    for(size_t i=0;i<count;i++) {
        execute("BEGIN IMMEDIATE; DELETE FROM doc_map");
        job_sql("INSERT INTO jobs SELECT * FROM incoming.jobs WHERE id=?1",ids[i]);
        job_sql("INSERT INTO collectors SELECT * FROM incoming.collectors WHERE job_id=?1",ids[i]);
        job_sql("INSERT INTO documents(job_id,kind,source,ordinal,observed_ns,bytes,sha256,codec,data) SELECT job_id,kind,source,ordinal,observed_ns,bytes,sha256,codec,data FROM incoming.documents WHERE job_id=?1 ORDER BY id",ids[i]);
        job_sql("INSERT INTO doc_map SELECT i.id,d.id FROM incoming.documents i JOIN documents d ON d.job_id=i.job_id AND d.kind=i.kind AND d.source=i.source AND d.ordinal=i.ordinal WHERE i.job_id=?1",ids[i]);
        job_sql("INSERT INTO samples(job_id,document_id,metric,observed_ns,entity,dimensions_json,value_json,value_number,value_integer) SELECT s.job_id,m.new_id,s.metric,s.observed_ns,s.entity,s.dimensions_json,s.value_json,s.value_number,s.value_integer FROM incoming.samples s LEFT JOIN doc_map m ON m.old_id=s.document_id WHERE s.job_id=?1",ids[i]);
        job_sql("INSERT INTO records(job_id,document_id,row_number,kind,dimensions_json,data_json,valid) SELECT r.job_id,m.new_id,r.row_number,r.kind,r.dimensions_json,r.data_json,r.valid FROM incoming.records r JOIN doc_map m ON m.old_id=r.document_id WHERE r.job_id=?1",ids[i]);
        index_copied();provenance(ids[i],argv[2],argv[3]);
        execute("COMMIT");printf("imported_job=%s source=%s\n",ids[i],argv[2]);fflush(stdout);free(ids[i]);
    }
    free(ids);
    q=prepare("PRAGMA foreign_key_check");if(sqlite3_step(q)!=SQLITE_DONE)fail("foreign key check");sqlite3_finalize(q);
    printf("imported_jobs=%zu source=%s\n",count,argv[2]);sqlite3_close(db);return 0;
}
