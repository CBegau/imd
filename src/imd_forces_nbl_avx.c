/******************************************************************************
 *
 * IMD -- The ITAP Molecular Dynamics Program
 *
 * Copyright 1996-2011 Institute for Theoretical and Applied Physics,
 * University of Stuttgart, D-70550 Stuttgart
 *
 ******************************************************************************/

/******************************************************************************
 *
 * imd_forces_nbl.c -- force loop with neighbor lists
 * optimized for parallelization & vectorization
 * This implementation is very memory intensive
 *
 ******************************************************************************/

/******************************************************************************
 * $Revision$
 * $Date$
 ******************************************************************************/

#include "imd.h"
#include "potaccess.h"

#define NBLMINLEN 10000

void estimate_nblist_size_pairs(int*);

int **cell_i = NULL;
int **cell_j = NULL;
int **num_i = NULL;
int **num_j = NULL;
int *pairsListLengths = NULL;
int *pairsListMaxLengths = NULL;
double *cutoffRadii = NULL;
int initialized = 0;

real *epot = NULL;
real *grad = NULL;
real *r2   = NULL;
int r2ListSize = 0;

void init(void){
	int i,j,m;
	int n = ntypes*ntypes;
	//Allocate the list of the number of pairs
	pairsListLengths = malloc(n* sizeof *pairsListLengths);
	pairsListMaxLengths = malloc(n* sizeof *pairsListMaxLengths);
	for (i=0; i<n; i++){
		pairsListLengths[i] = 0;
		pairsListMaxLengths[i] = 0;
	}

	//Compute/read the individual cut-off radii for the different pairs
	//if different cut-off radii are used for individual steps, use the maximal value
	cutoffRadii = malloc(n * sizeof *cutoffRadii);
	for (i=0; i<ntypes;i++){
		for (j=0; j<ntypes;j++){
			m = i*ntypes + j;
			cutoffRadii[m] = pair_pot.end[m];
#ifdef EAM2
			cutoffRadii[m] = MAX(cutoffRadii[m],rho_h_tab.end[m]);
#endif
		}
	}

	cell_i = malloc(n * sizeof *cell_i);
	cell_j = malloc(n * sizeof *cell_j);
	num_i  = malloc(n * sizeof *num_i);
	num_j  = malloc(n * sizeof *num_j);

	for (i=0; i<n; i++){
		cell_i[i] = NULL;
		cell_j[i] = NULL;
		num_i [i] = NULL;
		num_j [i] = NULL;
	}

	initialized = 1;
}

/******************************************************************************
 *
 *  deallocate (largest part of) neighbor list
 *
 ******************************************************************************/

void deallocate_nblist(void){
	if (!initialized) return;
	int i;
	for (i=0; i<ntypes * ntypes;i++){
		free(cell_i[i]);
		free(cell_j[i]);
		free(num_i[i]);
		free(num_j[i]);
	}

	free(cell_i);
	free(cell_j);
	free(num_i);
	free(num_j);
	free(pairsListLengths);
	free(cutoffRadii);

	if(r2ListSize != 0){
		free(epot);
		free(grad);
		free(r2);
		r2ListSize = 0;
	}


	have_valid_nbl = 0;
	initialized = 0;
}

/******************************************************************************
 *
 *  estimate_nblist_size
 *
 ******************************************************************************/

void estimate_nblist_size_pairs(int *pairs){
	int c, n;

	for (c=0; c<ntypes*ntypes; c++)
		pairs[c] = 0;

	/* for all cells */
	for (c = 0; c < ncells2; c++) {

		int i, c1 = cnbrs[c].np;
		cell *p = cell_array + c1;

		/* for each atom in cell */
		for (i = 0; i < p->n; i++) {

			int m;
			vektor d1;

			d1.x = ORT(p, i, X);
			d1.y = ORT(p, i, Y);
			d1.z = ORT(p, i, Z);
			int is = SORTE(p,i);

			/* for each neighboring atom */
			for (m = 0; m < NNBCELL; m++) {

				int c2, jstart, j;
				real r2;
				cell *q;

				c2 = cnbrs[c].nq[m];
				if (c2 < 0) continue;
				if (c2 == c1) jstart = i + 1;
				else jstart = 0;

				q = cell_array + c2;

				for (j = jstart; j < q->n; j++) {
					vektor d;
					d.x = ORT(q,j,X) - d1.x;
					d.y = ORT(q,j,Y) - d1.y;
					d.z = ORT(q,j,Z) - d1.z;

					int js = SORTE(q,j);

					r2 = SPROD(d, d);
					n = is*ntypes + js;
					if (r2 <= cutoffRadii[n]) pairs[n]++;
				}
			}
		}
	}
}

/******************************************************************************
 *
 *  make_nblist
 *
 ******************************************************************************/

void make_nblist(void){
	int i,j, k, n;

	/* update reference positions */
	for (k = 0; k < ncells; k++) {
		cell *p = cell_array + cnbrs[k].np;
		const int n = p->n;
		for (i = 0; i < n; i++) {
			NBL_POS(p, i, X) = ORT(p, i, X);
			NBL_POS(p, i, Y) = ORT(p, i, Y);
			NBL_POS(p, i, Z) = ORT(p, i, Z);
		}
	}

	n = ntypes*ntypes;


	estimate_nblist_size_pairs(pairsListLengths);

	//(re-allocate) pair lists
	for (j = 0; j<n; j++){
		int size = MAX( (int)(nbl_size * pairsListLengths[j]), NBLMINLEN);
		if( size > pairsListMaxLengths[j]){
			pairsListMaxLengths[j] = size;
			if (cell_i[j]) free(cell_i[j]);
			if (cell_j[j]) free(cell_j[j]);
			if (num_i[j]) free(num_i[j]);
			if (num_j[j]) free(num_j[j]);

			cell_i[j] = malloc(size * sizeof *cell_i[j]);
			cell_j[j] = malloc(size * sizeof *cell_j[j]);
			num_i[j]  = malloc(size * sizeof *num_i[j]);
			num_j[j]  = malloc(size * sizeof *num_j[j]);

			if (cell_i[j]==NULL || cell_j[j] == NULL || num_i[j] == NULL || num_j[j] == NULL){
				error("Cannot allocate neighbor pair list");
			}
		}
		//Set the number of used entries in the list to 0
		pairsListLengths[j] = 0;
	}


	/* for all cells */
	int c;
	for (c = 0; c < ncells2; c++) {
		int i, c1 = cnbrs[c].np;
		cell *p = cell_array + c1;

		/* for each atom in cell */
		for (i = 0; i < p->n; i++) {
			int m;
			vektor d1;

			d1.x = ORT(p, i, X);
			d1.y = ORT(p, i, Y);
			d1.z = ORT(p, i, Z);
			int is = SORTE(p,i);

			/* for each neighboring atom */
			for (m = 0; m < NNBCELL; m++) {

				int c2, jstart, j;
				real r2;
				cell *q;

				c2 = cnbrs[c].nq[m];
				if (c2 < 0) continue;
				if (c2 == c1) jstart = i + 1;
				else jstart = 0;

				q = cell_array + c2;

				for (j = jstart; j < q->n; j++) {
					vektor d;
					d.x = ORT(q,j,X) - d1.x;
					d.y = ORT(q,j,Y) - d1.y;
					d.z = ORT(q,j,Z) - d1.z;

					int js = SORTE(q,j);

					r2 = SPROD(d, d);
					n = is*ntypes + js;
					if (r2 <= cutoffRadii[n]) {
						k = pairsListLengths[n]++;
						cell_i[n][k] = c1;
						cell_j[n][k] = c2;
						num_i[n][k] = i;
						num_j[n][k] = j;
					}
				}
			}
		}
	}

	have_valid_nbl = 1;
	nbl_count++;
}

/******************************************************************************
 *
 *  calc_forces
 *
 ******************************************************************************/

void calc_forces(int steps){
	int i,j,k,n;
	int nPairs = ntypes*ntypes;

	if (!initialized){
		init();
		initialized = 1;
	}

	if (0 == have_valid_nbl) {
#ifdef MPI
		/* check message buffer size */
		if (0 == nbl_count % BUFSTEP) setup_buffers();
#endif
		/* update cell decomposition */
		fix_cells();
	}

	/* fill the buffer cells */
	send_cells(copy_cell, pack_cell, unpack_cell);

	/* make new neighbor lists */
	if (0 == have_valid_nbl) make_nblist();


	/* clear global accumulation variables */
	tot_pot_energy = 0.0;

	virial = 0.0;
	vir_xx = 0.0;
	vir_yy = 0.0;
	vir_xy = 0.0;
	vir_zz = 0.0;
	vir_yz = 0.0;
	vir_zx = 0.0;
	nfc++;

	/* clear per atom accumulation variables, also in buffer cells */
	for (k = 0; k < nallcells; k++) {
		cell *p = cell_array + k;
		const int n = p->n;
		for (i = 0; i < n; i++) {
			KRAFT(p,i,X) = 0.0;
			KRAFT(p,i,Y) = 0.0;
			KRAFT(p,i,Z) = 0.0;
			POTENG(p,i)  = 0.0;
#ifdef EAM2
			EAM_RHO(p,i) = 0.0;
#endif
#if defined(STRESS_TENS)
			PRESSTENS(p,i,xx) = 0.0;
			PRESSTENS(p,i,yy) = 0.0;
			PRESSTENS(p,i,xy) = 0.0;
			PRESSTENS(p,i,zz) = 0.0;
			PRESSTENS(p,i,yz) = 0.0;
			PRESSTENS(p,i,zx) = 0.0;
#endif
		}
	}

	/* clear total forces */
#ifdef RIGID
	if ( nsuperatoms>0 )
	for(i=0; i<nsuperatoms; i++) {
		superforce[i].x = 0.0;
		superforce[i].y = 0.0;
		superforce[i].z = 0.0;
	}
#endif


	int maxList = 0;
	for (i = 0; i<nPairs; i++)
		maxList = MAX(pairsListLengths[i],maxList);

	if(maxList>r2ListSize){
		r2ListSize = (int)(nbl_size*maxList);
		if(grad) free(grad);
		if(epot) free(epot);
		if(r2) free(r2);
		epot = malloc(r2ListSize * sizeof *epot);
		grad = malloc(r2ListSize * sizeof *grad);
		r2   = malloc(r2ListSize * sizeof *r2);
	}

	for (n = 0; n<nPairs; n++){
		const int* restrict cell_i_n = cell_i[n];
		const int* restrict cell_j_n = cell_j[n];
		const int* restrict num_i_n = num_i[n];
		const int* restrict num_j_n = num_j[n];

		const int m = pairsListLengths[n];

		const real potBegin = pair_pot.begin[n];
		const real potEnd = pair_pot.end[n];
		const real potEndPlus = pair_pot.end[n]+0.1;

		//Precompute distances
#ifdef INTEL_SIMD
#pragma ivdep
#endif
		for (i=0; i<m; i++){
			vektor v;
			cell *p = cell_array+cell_i_n[i];
			cell *q = cell_array+cell_j_n[i];
			v.x = ORT(q, num_j_n[i], X) - ORT(p, num_i_n[i], X);
			v.y = ORT(q, num_j_n[i], Y) - ORT(p, num_i_n[i], Y);
			v.z = ORT(q, num_j_n[i], Z) - ORT(p, num_i_n[i], Z);

			real r = SPROD(v,v);
			r2[i] = MIN(potEndPlus, r);
		}
#ifdef INTEL_SIMD
#pragma ivdep
#endif
		for (i=0; i<m; i++){
			real r = MAX( 0.0, r2[i] - potBegin);
			PAIR_INT_VEC(epot[i], grad[i], pair_pot, n, 1, r);
		}

		for (i=0; i<m; i++){
			vektor v, force;

			if (r2[i] <= potEnd){
				cell *p = cell_array+cell_i_n[i];
				cell *q = cell_array+cell_j_n[i];
				v.x = ORT(q, num_j_n[i], X) - ORT(p, num_i_n[i], X);
				v.y = ORT(q, num_j_n[i], Y) - ORT(p, num_i_n[i], Y);
				v.z = ORT(q, num_j_n[i], Z) - ORT(p, num_i_n[i], Z);

				force.x = v.x * grad[i];
				force.y = v.y * grad[i];
				force.z = v.z * grad[i];

				KRAFT(q, num_j_n[i],X) -= force.x;
				KRAFT(q, num_j_n[i],Y) -= force.y;
				KRAFT(q, num_j_n[i],Z) -= force.z;

				KRAFT(p, num_i_n[i],X) += force.x;
				KRAFT(p, num_i_n[i],Y) += force.y;
				KRAFT(p, num_i_n[i],Z) += force.z;

				POTENG(p, num_i_n[i]) += epot[i] * 0.5;
				POTENG(q, num_j_n[i]) += epot[i] * 0.5;

#ifdef P_AXIAL
				vir_xx -= v.x * force.x;
				vir_yy -= v.y * force.y;
				vir_zz -= v.z * force.z;
#else
				virial -= r2[i] * grad[i];
#endif

#ifdef STRESS_TENS
				if (do_press_calc) {
					/* avoid double counting of the virial */
					force.x *= 0.5;
					force.y *= 0.5;
					force.z *= 0.5;

					PRESSTENS(p, num_i_n[i],xx) -= v.x * force.x;
					PRESSTENS(q, num_j_n[i],xx) -= v.x * force.x;
					PRESSTENS(p, num_i_n[i],yy) -= v.y * force.y;
					PRESSTENS(q, num_j_n[i],yy) -= v.y * force.y;
					PRESSTENS(p, num_i_n[i],xy) -= v.x * force.y;
					PRESSTENS(q, num_j_n[i],xy) -= v.x * force.y;
					PRESSTENS(p, num_i_n[i],zz) -= v.z * force.z;
					PRESSTENS(q, num_j_n[i],zz) -= v.z * force.z;
					PRESSTENS(p, num_i_n[i],yz) -= v.y * force.z;
					PRESSTENS(q, num_j_n[i],yz) -= v.y * force.z;
					PRESSTENS(p, num_i_n[i],zx) -= v.z * force.x;
					PRESSTENS(q, num_j_n[i],zx) -= v.z * force.x;
				}
#endif
			}
		}
#ifdef EAM2

#endif

	}// pairs n


	//Sum total potential energy
	for (k=0; k<nallcells; k++) {
		cell *p = cell_array+k;
		int i;
		for (i=0; i<p->n; i++)
			tot_pot_energy += POTENG(p,i);
	}

//	/* pair interactions - for all atoms */
//	n = 0;
//	for (k = 0; k < ncells; k++) {
//		cell *p = cell_array + cnbrs[k].np;
//		for (i = 0; i < p->n; i++) {
//
//#ifdef STRESS_TENS
//			sym_tensor pp = {0.0,0.0,0.0,0.0,0.0,0.0};
//#endif
//
//			vektor d1, ff = {0.0, 0.0, 0.0};
//			real ee = 0.0;
//			real eam_r = 0.0, eam_p = 0.0;
//			int m, it, nb = 0;
//
//			d1.x = ORT(p, i, X);
//			d1.y = ORT(p, i, Y);
//			d1.z = ORT(p, i, Z);
//
//			it = SORTE(p, i);
//
//			/* loop over neighbors */
//			for (m = tl[n]; m < tl[n + 1]; m++) {
//
//				vektor d, force;
//				cell *q;
//				real pot, grad, r2, rho_h;
//				int c, j, jt, col, col2, inc = ntypes * ntypes;
//
//				c = cl_num[tb[m]];
//				j = tb[m] - cl_off[c];
//				q = cell_array + c;
//
//				d.x = ORT(q,j,X) - d1.x;
//				d.y = ORT(q,j,Y) - d1.y;
//				d.z = ORT(q,j,Z) - d1.z;
//
//				r2 = SPROD(d, d);
//				jt = SORTE(q, j);
//				col = it * ntypes + jt;
//				col2 = jt * ntypes + it;
//
//				/* compute pair interactions */
//				if (r2 <= pair_pot.end[col]) {
//					PAIR_INT(pot, grad, pair_pot, col, inc, r2, is_short);
//
//					tot_pot_energy += pot;
//					force.x = d.x * grad;
//					force.y = d.y * grad;
//					force.z = d.z * grad;
//
//					KRAFT(q,j,X) -= force.x;
//					KRAFT(q,j,Y) -= force.y;
//					KRAFT(q,j,Z) -= force.z;
//
//					ff.x += force.x;
//					ff.y += force.y;
//					ff.z += force.z;
//
//					pot *= 0.5; /* avoid double counting */
//					ee += pot;
//					POTENG(q,j) += pot;
//#ifdef P_AXIAL
//					vir_xx -= d.x * force.x;
//					vir_yy -= d.y * force.y;
//					vir_zz -= d.z * force.z;
//#else
//					virial -= r2 * grad;
//#endif
//
//#ifdef STRESS_TENS
//					if (do_press_calc) {
//						/* avoid double counting of the virial */
//						force.x *= 0.5;
//						force.y *= 0.5;
//						force.z *= 0.5;
//
//						pp.xx -= d.x * force.x;
//						PRESSTENS(q,j,xx) -= d.x * force.x;
//						pp.yy -= d.y * force.y;
//						PRESSTENS(q,j,yy) -= d.y * force.y;
//						pp.xy -= d.x * force.y;
//						PRESSTENS(q,j,xy) -= d.x * force.y;
//						pp.zz -= d.z * force.z;
//						PRESSTENS(q,j,zz) -= d.z * force.z;
//						pp.yz -= d.y * force.z;
//						PRESSTENS(q,j,yz) -= d.y * force.z;
//						pp.zx -= d.z * force.x;
//						PRESSTENS(q,j,zx) -= d.z * force.x;
//					}
//#endif
//				}
//
//#ifdef EAM2
//				/* compute host electron density */
//				if (r2 < rho_h_tab.end[col]) {
//					VAL_FUNC(rho_h, rho_h_tab, col, inc, r2, is_short);
//					eam_r += rho_h;
//				}
//				if (it==jt) {
//					if (r2 < rho_h_tab.end[col]) {
//						EAM_RHO(q,j) += rho_h;
//					}
//				} else {
//					if (r2 < rho_h_tab.end[col2]) {
//						VAL_FUNC(rho_h, rho_h_tab, col2, inc, r2, is_short);
//						EAM_RHO(q,j) += rho_h;
//					}
//				}
//#endif
//
//			}
//			KRAFT(p,i,X) += ff.x;
//			KRAFT(p,i,Y) += ff.y;
//			KRAFT(p,i,Z) += ff.z;
//
//			POTENG(p,i) += ee;
//
//#ifdef EAM2
//			EAM_RHO(p,i) += eam_r;
//#endif
//
//#ifdef STRESS_TENS
//			if (do_press_calc) {
//				PRESSTENS(p,i,xx) += pp.xx;
//				PRESSTENS(p,i,yy) += pp.yy;
//				PRESSTENS(p,i,xy) += pp.xy;
//				PRESSTENS(p,i,zz) += pp.zz;
//				PRESSTENS(p,i,yz) += pp.yz;
//				PRESSTENS(p,i,zx) += pp.zx;
//			}
//#endif
//			n++;
//		}
//	}
//	if (is_short) fprintf(stderr, "Short distance, pair, step %d!\n", steps);
//
//#ifdef EAM2
//
//	/* collect host electron density */
//	send_forces(add_rho,pack_rho,unpack_add_rho);
//
//	/* compute embedding energy and its derivative */
//	for (k=0; k<ncells; k++) {
//		cell *p = CELLPTR(k);
//		real pot, tmp, tr;
//
//		for (i=0; i<p->n; i++) {
//			PAIR_INT( pot, EAM_DF(p,i), embed_pot, SORTE(p,i),
//					ntypes, EAM_RHO(p,i), idummy);
//			POTENG(p,i) += pot;
//			tot_pot_energy += pot;
//		}
//	}
//
//	/* distribute derivative of embedding energy */
//	send_cells(copy_dF,pack_dF,unpack_dF);
//
//	/* EAM interactions - for all atoms */
//	n=0;
//	for (k=0; k<ncells; k++) {
//		cell *p = CELLPTR(k);
//		for (i=0; i<p->n; i++) {
//
//#ifdef STRESS_TENS
//			sym_tensor pp = {0.0,0.0,0.0,0.0,0.0,0.0};
//#endif
//			vektor d1, ff = {0.0,0.0,0.0};
//			int m, it;
//
//			d1.x = ORT(p,i,X);
//			d1.y = ORT(p,i,Y);
//			d1.z = ORT(p,i,Z);
//
//			it = SORTE(p,i);
//
//			/* loop over neighbors */
//			for (m=tl[n]; m<tl[n+1]; m++) {
//
//				vektor d, force = {0.0,0.0,0.0};
//				real r2;
//				int c, j, jt, col1, col2, inc = ntypes * ntypes, have_force=0;
//				cell *q;
//
//				c = cl_num[ tb[m] ];
//				j = tb[m] - cl_off[c];
//				q = cell_array + c;
//
//				d.x = ORT(q,j,X) - d1.x;
//				d.y = ORT(q,j,Y) - d1.y;
//				d.z = ORT(q,j,Z) - d1.z;
//				r2 = SPROD(d,d);
//				jt = SORTE(q,j);
//				col1 = jt * ntypes + it;
//				col2 = it * ntypes + jt;
//
//				if ((r2 < rho_h_tab.end[col1]) || (r2 < rho_h_tab.end[col2])) {
//
//					real pot, grad, rho_i_strich, rho_j_strich, rho_i, rho_j;
//
//					/* take care: particle i gets its rho from particle j.    */
//					/* This is tabulated in column it*ntypes+jt.              */
//					/* Here we need the giving part from column jt*ntypes+it. */
//
//					/* rho_strich_i(r_ij) */
//					/* rho_strich_i(r_ij) and rho_i(r_ij) */
//					PAIR_INT(rho_i, rho_i_strich, rho_h_tab, col1, inc, r2, is_short);
//
//					/* rho_strich_j(r_ij) */
//					if (col1==col2) {
//						rho_j_strich = rho_i_strich;
//					} else {
//						DERIV_FUNC(rho_j_strich, rho_h_tab, col2, inc, r2, is_short);
//					}
//
//					/* put together (dF_i and dF_j are by 0.5 too big) */
//					grad = 0.5 * (EAM_DF(p,i)*rho_j_strich + EAM_DF(q,j)*rho_i_strich);
//
//					/* store force in temporary variable */
//					force.x = d.x * grad;
//					force.y = d.y * grad;
//					force.z = d.z * grad;
//					have_force=1;
//				}
//
//				/* accumulate forces */
//				if (have_force) {
//					KRAFT(q,j,X) -= force.x;
//					KRAFT(q,j,Y) -= force.y;
//					KRAFT(q,j,Z) -= force.z;
//					ff.x += force.x;
//					ff.y += force.y;
//					ff.z += force.z;
//#ifdef P_AXIAL
//					vir_xx -= d.x * force.x;
//					vir_yy -= d.y * force.y;
//					vir_zz -= d.z * force.z;
//#else
//					virial -= SPROD(d,force);
//#endif
//
//#ifdef STRESS_TENS
//					if (do_press_calc) {
//						/* avoid double counting of the virial */
//						force.x *= 0.5;
//						force.y *= 0.5;
//						force.z *= 0.5;
//
//						pp.xx -= d.x * force.x;
//						pp.yy -= d.y * force.y;
//						pp.zz -= d.z * force.z;
//						pp.yz -= d.y * force.z;
//						pp.zx -= d.z * force.x;
//						pp.xy -= d.x * force.y;
//
//						PRESSTENS(q,j,xx) -= d.x * force.x;
//						PRESSTENS(q,j,yy) -= d.y * force.y;
//						PRESSTENS(q,j,zz) -= d.z * force.z;
//						PRESSTENS(q,j,yz) -= d.y * force.z;
//						PRESSTENS(q,j,zx) -= d.z * force.x;
//						PRESSTENS(q,j,xy) -= d.x * force.y;
//					}
//#endif
//				}
//			}
//			KRAFT(p,i,X) += ff.x;
//			KRAFT(p,i,Y) += ff.y;
//			KRAFT(p,i,Z) += ff.z;
//#ifdef STRESS_TENS
//			if (do_press_calc) {
//				PRESSTENS(p,i,xx) += pp.xx;
//				PRESSTENS(p,i,yy) += pp.yy;
//				PRESSTENS(p,i,zz) += pp.zz;
//				PRESSTENS(p,i,yz) += pp.yz;
//				PRESSTENS(p,i,zx) += pp.zx;
//				PRESSTENS(p,i,xy) += pp.xy;
//			}
//#endif
//			n++;
//		}
//	}
//	if (is_short) fprintf(stderr, "\n Short distance, EAM, step %d!\n",steps);
//
//#endif /* EAM2 */

#ifdef MPI
	real tmpvec1[8], tmpvec2[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	/* sum up results of different CPUs */
	tmpvec1[0] = tot_pot_energy;
	tmpvec1[1] = virial;
	tmpvec1[2] = vir_xx;
	tmpvec1[3] = vir_yy;
	tmpvec1[4] = vir_zz;
	tmpvec1[5] = vir_xy;
	tmpvec1[6] = vir_yz;
	tmpvec1[7] = vir_zx;
	MPI_Allreduce( tmpvec1, tmpvec2, 8, REAL, MPI_SUM, cpugrid);
	tot_pot_energy = tmpvec2[0];
	virial = tmpvec2[1];
	vir_xx = tmpvec2[2];
	vir_yy = tmpvec2[3];
	vir_zz = tmpvec2[4];
	vir_xy = tmpvec2[5];
	vir_yz = tmpvec2[6];
	vir_zx = tmpvec2[7];
#endif

	/* add forces back to original cells/cpus */
	send_forces(add_forces, pack_forces, unpack_forces);

}

/******************************************************************************
 *
 *  check_nblist
 *
 ******************************************************************************/

void check_nblist(){
	real r2, max1 = 0.0, max2;
	vektor d;
	int k;

	/* compare with reference positions */
	for (k = 0; k < NCELLS; k++) {
		int i;
		cell *p = CELLPTR(k);
		for (i = 0; i < p->n; i++) {
			d.x = ORT(p,i,X) - NBL_POS(p, i, X);
			d.y = ORT(p,i,Y) - NBL_POS(p, i, Y);
			d.z = ORT(p,i,Z) - NBL_POS(p, i, Z);

			r2 = SPROD(d, d);
			if (r2 > max1) max1 = r2;
		}
	}

#ifdef MPI
	MPI_Allreduce( &max1, &max2, 1, REAL, MPI_MAX, cpugrid);
#else
	max2 = max1;
#endif
	if (max2 > SQR(0.5 * nbl_margin)) have_valid_nbl = 0;
}

