#ifndef FLASH_HOLD_SELECTION_H
#define FLASH_HOLD_SELECTION_H
/* ---- the split: which rows (or experts) of a tensor this die keeps ---- */
typedef struct { int r0, n; } range_t;
typedef struct { range_t r[6]; int nr; int rows_total; int experts; unsigned row_grain,embedding_first,embedding_groups; } sel_t;   /* rows_total: rows the ranges index */
static int split_on;
static int balance_on;
static int unique_on;
static split_t SP;
#include "balance/layout.h"
#include "balance/embedding.h"
#include "unique/layout.h"
static void holder_split(int die,int dies,int block) {
    if(balance_on && unique_on)unique_split(&SP,(unsigned)die,(unsigned)dies);
    else if(balance_on)balance_split(&SP,(unsigned)die,(unsigned)dies);
    else split_of(&SP,die,dies,block);
}
static void sel_whole(sel_t *s, int rows, int experts) { s->nr = 1; s->r[0].r0 = 0; s->r[0].n = experts > 1 ? experts : rows; s->rows_total = rows; s->experts = experts; s->row_grain=s->embedding_first=s->embedding_groups=0; }
static void select_rows(const char *name, int rows, int experts, sel_t *s) {
	sel_whole(s, rows, experts);
	if (!split_on) return;
	const split_t *p = &SP;
    if(balance_on && !strcmp(name,"token_embd.weight")) {
        s->embedding_first=(unsigned)p->cb0;s->embedding_groups=(unsigned)p->cbn;return;
    }
	/* the head: output.weight rows, output_hc_down rows, output_hc_up the 4 lanes of the z columns; token_embd whole */
	if (!strcmp(name, "output.weight")) { s->nr = 1; s->r[0] = (range_t){ p->o0, p->on }; return; }
	if (!strcmp(name, "output_hc_down.weight")) { s->nr = 1; s->r[0] = (range_t){ p->hd0, p->hdn }; return; }
	if (!strcmp(name, "output_hc_up.weight")) { s->nr = 4; for (int l = 0; l < 4; l++) s->r[l] = (range_t){ 2560 * l + 32 * p->cb0, 32 * p->cbn }; return; }
	const char *x = suffix(name);
	if (!x) return;
    if(balance_on && unique_on) {
        if(!strcmp(x,"ffn_gate_inp_x.weight")) {
            s->nr=1;s->r[0]=(range_t){(int)unique_router_first((unsigned)p->d,(unsigned)p->n),
                                    (int)unique_router_rows((unsigned)p->d,(unsigned)p->n)};return;
        }
        if(!strcmp(x,"attn_k.weight") || !strcmp(x,"attn_v.weight")) {
            s->row_grain=FLASH_BALANCE_EXPERT_GRAIN;return;
        }
        if(!strcmp(x,"attn_qkv.weight") || !strcmp(x,"ssm_conv1d.weight")) {
            s->nr=0;
            s->r[s->nr++]=(range_t){128*p->klo,128*p->nk};
            s->r[s->nr++]=(range_t){2048+128*p->klo,128*p->nk};
            for(int group=0;group<3;group++)s->r[s->nr++]=(range_t){4096+128*(p->klo+16*group),128*p->nk};
            return;
        }
        if(!strcmp(x,"attn_gate.weight")) {
            s->nr=3;for(int group=0;group<3;group++)s->r[group]=(range_t){128*(p->klo+16*group),128*p->nk};return;
        }
        if(!strcmp(x,"ssm_ab.weight") || !strcmp(x,"ssm_ab.bias")) {
            s->nr=0;s->rows_total=96;
            for(int side=0;side<2;side++)for(int group=0;group<3;group++)
                s->r[s->nr++]=(range_t){48*side+p->klo+16*group,p->nk};
            return;
        }
        if(!strcmp(x,"ssm_a")) {
            s->nr=3;s->rows_total=48;
            for(int group=0;group<3;group++)s->r[group]=(range_t){p->klo+16*group,p->nk};
            return;
        }
    }
    if(balance_on) {
        if(strstr(x,"_exps.weight") || strstr(x,"_shexp.weight")) {
            s->row_grain=FLASH_BALANCE_EXPERT_GRAIN;return;
        }
        if(!strcmp(x,"attn_q.weight")) {
            s->nr=1;s->r[0]=(range_t){512*(int)balance_attention_first((unsigned)SP.d,(unsigned)SP.n),512*(int)balance_attention_heads((unsigned)SP.d,(unsigned)SP.n)};return;
        }
        /* Four query heads consume one of the two GQA K/V heads. The balanced
         * cache stores only that head. Indexer projection rows are striped and
         * published to every die before local indexer normalization/selection. */
        if(!strcmp(x,"attn_k.weight") || !strcmp(x,"attn_v.weight")) {
            s->nr=1;s->r[0]=(range_t){256*(int)balance_attention_kv_head((unsigned)SP.d,(unsigned)SP.n),256};return;
        }
        if(!strcmp(x,"indexer_x.weight")) {s->row_grain=FLASH_BALANCE_EXPERT_GRAIN;return;}
        if(!strcmp(x,"indexer.q_norm.weight") || !strcmp(x,"indexer.k_norm.weight"))return;
    }
	/* The draft block's final HC read uses the ordinary output-row split. */
	if (!strcmp(x, "nextn.hc_head_down.weight")) { s->nr = 1; s->r[0] = (range_t){ p->hd0, p->hdn }; return; }
	if (!strcmp(x, "nextn.hc_head_up.weight")) { s->nr = 4; for (int l = 0; l < 4; l++) s->r[l] = (range_t){ 2560 * l + 32 * p->cb0, 32 * p->cbn }; return; }
	/* the shared expert runs on one die per block */
	if (strstr(x, "_shexp.weight") && !p->shexp) { s->nr = 1; s->r[0] = (range_t){ 0, 0 }; return; }
	if (!strcmp(x, "hc_attn_dj.weight") || !strcmp(x, "hc_ffn_dj.weight")) { s->nr = 1; s->r[0] = (range_t){ p->dj0, p->djn }; }
	else if (!strcmp(x, "hc_attn_up.weight") || !strcmp(x, "hc_ffn_up.weight")) { s->nr = 4; for (int l = 0; l < 4; l++) s->r[l] = (range_t){ 2560 * l + 32 * p->cb0, 32 * p->cbn }; }
	else if (!strcmp(x, "attn_qkv.weight") || !strcmp(x, "ssm_conv1d.weight")) {
		/* key heads (klo + j) % 16, j < nk: one range, or two when it wraps past head 15 */
		int n1 = p->klo + p->nk <= 16 ? p->nk : 16 - p->klo, n2 = p->nk - n1; s->nr = 0;
		s->r[s->nr++] = (range_t){ 128 * p->klo, 128 * n1 }; if (n2) s->r[s->nr++] = (range_t){ 0, 128 * n2 };
		s->r[s->nr++] = (range_t){ 2048 + 128 * p->klo, 128 * n1 }; if (n2) s->r[s->nr++] = (range_t){ 2048, 128 * n2 };
		s->r[s->nr++] = (range_t){ 4096 + 128 * p->h0, 128 * p->hn };
	}
	else if (!strcmp(x, "attn_gate.weight")) { s->nr = 1; s->r[0] = (range_t){ 128 * p->h0, 128 * p->hn }; }
	else if (!strcmp(x, "ssm_ab.weight") || !strcmp(x, "ssm_ab.bias")) { s->nr = 2; s->r[0] = (range_t){ p->h0, p->hn }; s->r[1] = (range_t){ 48 + p->h0, p->hn }; s->rows_total = 96; }
	else if (!strcmp(x, "ssm_a")) { s->nr = 1; s->r[0] = (range_t){ p->h0, p->hn }; s->rows_total = 48; }
	else if (!strcmp(x, "ssm_out.weight")) { s->nr = 1; s->r[0] = (range_t){ p->y0, p->yn }; }
	/* the attention proper (q/k/v, the indexer projections and norms) runs on the die of the layer (ATTN_MAP), the output projection is row-split */
	else if (!strcmp(x, "attn_q.weight") || !strcmp(x, "attn_k.weight") || !strcmp(x, "attn_v.weight") || !strcmp(x, "indexer_x.weight") || !strcmp(x, "indexer.q_norm.weight") || !strcmp(x, "indexer.k_norm.weight")) { s->nr = 1; s->r[0] = (range_t){ 0, p->d == attn_die_of(atoi(name + 4)) ? rows : 0 }; }
	else if (!strcmp(x, "attn_output.weight")) { s->nr = 1; s->r[0] = (range_t){ p->y0, p->yn }; }
	else if (!strcmp(x, "ple_key.weight")) { s->nr = 1; s->r[0] = (range_t){ p->pk0, p->pkn }; }
	else if (!strcmp(x, "ple_value.weight")) { s->nr = 1; s->r[0] = (range_t){ p->y0, p->yn }; }
	else if (!strcmp(x, "ffn_gate_exps.weight") || !strcmp(x, "ffn_up_exps.weight") || !strcmp(x, "ffn_down_exps.weight")) { s->nr = 1; s->r[0] = (range_t){ p->e0, p->en }; }
}

#endif
