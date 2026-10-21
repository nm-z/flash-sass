/* Export only selected benchmark stdout/stderr from the canonical GPUQ DB.
 * Identity/environment and unrelated collector documents stay in the private DB.
 * Build: cc -O2 -Wall -Wextra -Werror export_logs.c -lsqlite3 -lz -o export_logs
 * Usage: ./export_logs history.sqlite3 jobs.json > stdout-stderr.jsonl
 */
#include <sqlite3.h>
#include <zlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *s) { fprintf(stderr,"export_logs: %s\n",s);exit(1); }
static void quoted(const unsigned char *s,size_t n) {
    putchar('"');
    for(size_t i=0;i<n;i++) {
        unsigned c=s[i];
        if(c=='"'||c=='\\') { putchar('\\');putchar((int)c); }
        else if(c<32)printf("\\u%04x",c);
        else putchar((int)c);
    }
    putchar('"');
}
int main(int argc,char **argv) {
    if(argc!=3)fail("usage: export_logs history.sqlite3 jobs.json");
    sqlite3 *db=NULL;sqlite3_stmt *q=NULL;
    if(sqlite3_open_v2(argv[1],&db,SQLITE_OPEN_READONLY,NULL)!=SQLITE_OK)fail("open archive");
    FILE *f=fopen(argv[2],"rb");if(!f)fail("open job list");
    if(fseek(f,0,SEEK_END))fail("seek job list");
    long size=ftell(f);if(size<0||size>8*1024*1024)fail("job list size");
    rewind(f);char *manifest=malloc((size_t)size+1);if(!manifest)fail("job list allocation");
    if(fread(manifest,1,(size_t)size,f)!=(size_t)size)fail("read job list");
    fclose(f);manifest[size]=0;
    const char *select_jobs="CREATE TEMP TABLE selected_jobs AS SELECT json_extract(value,'$.source_database') AS source_database,json_extract(value,'$.id') AS job_id FROM json_each(?1)";
    if(sqlite3_prepare_v2(db,select_jobs,-1,&q,NULL)!=SQLITE_OK)fail(sqlite3_errmsg(db));
    sqlite3_bind_text(q,1,manifest,(int)size,SQLITE_TRANSIENT);
    if(sqlite3_step(q)!=SQLITE_DONE)fail(sqlite3_errmsg(db));
    sqlite3_finalize(q);free(manifest);
    const char *sql="SELECT j.source_database,d.job_id,d.id,d.kind,d.codec,d.bytes,d.data FROM documents d JOIN selected_jobs j ON j.job_id=d.job_id WHERE d.kind IN ('stdout','stderr') ORDER BY j.source_database,d.job_id,d.observed_ns,d.id";
    if(sqlite3_prepare_v2(db,sql,-1,&q,NULL)!=SQLITE_OK)fail(sqlite3_errmsg(db));
    int rc;
    while((rc=sqlite3_step(q))==SQLITE_ROW) {
        sqlite3_int64 count=sqlite3_column_int64(q,5);
        if(count<0||count>128*1024*1024)fail("invalid document size");
        size_t n=(size_t)count;unsigned char *text=malloc(n+1);
        if(!text)fail("allocation");
        const unsigned char *blob=sqlite3_column_blob(q,6);
        int stored=sqlite3_column_bytes(q,6);
        const char *codec=(const char *)sqlite3_column_text(q,4);
        if(!strcmp(codec,"raw")) {
            if((size_t)stored!=n)fail("raw document size mismatch");
            if(n)memcpy(text,blob,n);
        } else if(!strcmp(codec,"zlib")) {
            uLongf out=(uLongf)(n+1);
            if(uncompress(text,&out,blob,(uLong)stored)!=Z_OK||out!=n)fail("zlib document mismatch");
        } else fail("unknown codec");
        fputs("{\"source_database\":",stdout);
        quoted(sqlite3_column_text(q,0),(size_t)sqlite3_column_bytes(q,0));
        fputs(",\"job_id\":",stdout);
        quoted(sqlite3_column_text(q,1),(size_t)sqlite3_column_bytes(q,1));
        printf(",\"document_id\":%lld,\"kind\":",(long long)sqlite3_column_int64(q,2));
        quoted(sqlite3_column_text(q,3),(size_t)sqlite3_column_bytes(q,3));
        fputs(",\"text\":",stdout);quoted(text,n);fputs("}\n",stdout);free(text);
    }
    if(rc!=SQLITE_DONE)fail(sqlite3_errmsg(db));
    sqlite3_finalize(q);sqlite3_close(db);
    if(fflush(stdout)||ferror(stdout))fail("output write");
    return 0;
}
