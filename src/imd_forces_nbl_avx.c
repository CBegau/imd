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

#define PAIR_INT_VEC(pot, grd, pt, col, inc, r2)                             \
{                                                                            \
  real r2a, istep, chi, p0, p1, p2, dv, d2v, *ptr;                           \
  int kk;                                                                   \
                                                                             \
  /* indices into potential table */                                         \
  istep = (pt).invstep[col];                                                 \
  r2a   = r2 * istep;                                                        \
  kk    = (int) (r2a);                                                       \
  chi   = r2a - kk;                                                          \
                                                                             \
  /* intermediate values */                                                  \
  ptr = PTR_2D((pt).table, (col),kk, (inc),(pt).maxsteps);                   \
  p0  = *ptr; ptr ++;                                                        \
  p1  = *ptr; ptr ++;                                                        \
  p2  = *ptr;                                                                \
  dv  = p1 - p0;                                                             \
  d2v = p2 - 2. * p1 + p0;                                                    \
                                                                             \
  /* potential and twice the derivative */                                   \
  pot = p0 + chi * dv + 0.5 * chi * (chi - 1.) * d2v;                         \
  grd = 2. * istep * (dv + (chi - 0.5) * d2v);                                \
}

#define NBLMINLEN 10000
struct ppair {
	int p[4];
};

struct ppair** restrict pairLists = NULL;

int* restrict pairsListLengths = NULL;
int* restrict pairsListMaxLengths = NULL;
real* restrict cutoffRadii = NULL;
int initialized = 0;

real* restrict epot = NULL;
real* restrict grad = NULL;
#ifdef EAM2
real* restrict rho_grad1 = NULL;
real* restrict rho_grad2 = NULL;
real* restrict rho1 = NULL;
real* restrict rho2 = NULL;
#endif
real* restrict r2 = NULL;
int r2ListSize = 0;
int epotListSize = 0;

void init(void){
	int i,j,m;
	int n = ntypes*ntypes;
	//Allocate the list of the number of pairs
	pairsListLengths = malloc(n* sizeof *pairsListLengths);
	pairsListMaxLengths = malloc(n* sizeof *pairsListMaxLengths);
	for (i=0; i<n; i++){
		pairsListLengths[i] = 0;
		pairsListMaxLengths[i] = NBLMINLEN;
	}

	//Compute/read the individual cut-off radii for the different pairs
	//if different cut-off radii are used for individual steps, use the maximal value
	cutoffRadii = malloc(n * sizeof *cutoffRadii);
	for (i=0; i<ntypes;i++){
		for (j=0; j<ntypes;j++){
			m = i*ntypes + j;
			cutoffRadii[m] = pair_pot.end[m]+nbl_margin;
#ifdef EAM2
			cutoffRadii[m] = MAX(cutoffRadii[m],rho_h_tab.end[m]+nbl_margin);
#endif
		}
	}

	pairLists = malloc(n * sizeof *pairLists);

	for (i=0; i<n; i++)
		pairLists[i] = malloc(pairsListMaxLengths[i] * sizeof *pairLists[i]);

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
	for (i=0; i<ntypes * ntypes;i++)
		free(pairLists[i]);

	free(pairLists);
	free(pairsListLengths);
	free(cutoffRadii);

	if(r2ListSize != 0){
		free(epot);
		free(grad);
		free(r2);
#ifdef EAM2
		free(rho_grad1);
		free(rho_grad2);
		free(rho1);
		free(rho2);
#endif
		r2ListSize = 0;
	}


	have_valid_nbl = 0;
	initialized = 0;
}

/******************************************************************************
 *
 *  make_nblist
 *
 ******************************************************************************/

void make_nblist(void){
	int i,j, k, n;

	/* update reference positions */
#ifdef OMP
#pragma omp parallel for
#endif
	for (k = 0; k < ncells; k++) {
		cell *p = cell_array + cnbrs[k].np;
		const int n = p->n;
#ifdef INTEL_SIMD
#pragma ivdep
#endif
		for (i = 0; i < n; i++) {
			NBL_POS(p, i, X) = ORT(p, i, X);
			NBL_POS(p, i, Y) = ORT(p, i, Y);
			NBL_POS(p, i, Z) = ORT(p, i, Z);
		}
	}

	n = ntypes*ntypes;

	//(re-allocate) pair lists
	for (j = 0; j<n; j++)
		pairsListLengths[j] = 0;

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
						pairLists[n][k].p[0] = c1;
						pairLists[n][k].p[1] = c2;
						pairLists[n][k].p[2] = i;
						pairLists[n][k].p[3] = j;
						//Double the size of the table, if running out of space
						if(pairsListLengths[n] == pairsListMaxLengths[n]){
							pairsListMaxLengths[n] *= 2;
							pairLists[n] = realloc(pairLists[n], pairsListMaxLengths[n]*sizeof *pairLists[n]);
						}
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
	const int nPairs = ntypes*ntypes;

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
#ifdef OMP
#pragma omp parallel for
#endif
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


	int sumList = 0;
	int longestList = 0;
	for (i = 0; i<nPairs; i++){
		sumList += pairsListLengths[i];
		longestList = MAX(longestList, pairsListLengths[i]);
	}

	if(sumList>r2ListSize){
		r2ListSize = (int)(nbl_size*sumList);
		if(r2) free(r2);
		if(grad) free(grad);
		if(epot) free(epot);
		
		r2   = malloc(r2ListSize * sizeof *r2);
		epot = malloc(r2ListSize * sizeof *epot);
		grad = malloc(r2ListSize * sizeof *grad);

#ifdef EAM2
		if(rho_grad1) free(rho_grad1);
		if(rho_grad2) free(rho_grad2);
		rho_grad1 = malloc(r2ListSize * sizeof *rho_grad1);
		rho_grad2 = malloc(r2ListSize * sizeof *rho_grad2);
#endif
	}

	if(longestList>epotListSize){
		epotListSize = (int)(nbl_size*longestList);
#ifdef EAM2
		if(rho1) free(rho1);
		if(rho2) free(rho2);
		rho1 = malloc(epotListSize * sizeof *rho1);
		rho2 = malloc(epotListSize * sizeof *rho2);
#endif
	}

	int cellpairOffset = 0;
	for (n = 0; n<nPairs; n++){
		const struct ppair* restrict pair = pairLists[n];

		const int m = pairsListLengths[n];

		const real potBegin = pair_pot.begin[n];
		const real potEnd = pair_pot.end[n];

		const int type1 = n / ntypes;
		const int type2 = n % ntypes;
		const int col1 = n;
		const int col2 = type2 * ntypes + type1;

#ifdef EAM2
		const real rhoBeginCol1 = rho_h_tab.begin[col1];
		const real rhoEndCol1 = rho_h_tab.end[col1];
		const real rhoBeginCol2 = rho_h_tab.begin[col2];
		const real rhoEndCol2 = rho_h_tab.end[col2];
#endif

		//Precompute squared distances and the pair-potential part
#ifdef INTEL_SIMD
#pragma ivdep
#endif
#ifdef OMP
#pragma omp parallel for
#endif
		for (i=0; i<m; i++){
			vektor v;
			cell *p = cell_array+pair[i].p[0];
			cell *q = cell_array+pair[i].p[1];
			int n_i = pair[i].p[2];
			int n_j = pair[i].p[3];
			//Compute squared distance
			v.x = ORT(q, n_j, X) - ORT(p, n_i, X);
			v.y = ORT(q, n_j, Y) - ORT(p, n_i, Y);
			v.z = ORT(q, n_j, Z) - ORT(p, n_i, Z);
			r2[i+cellpairOffset] = SPROD(v,v);

			//Clamp distance into the valid range of the potential table
			real r = MIN(potEnd, r2[i+cellpairOffset]);
			r = MAX( 0.0, r - potBegin);
			//Compute pair potential and store epot and its gradient
			PAIR_INT_VEC(epot[i+cellpairOffset], grad[i+cellpairOffset], pair_pot, col1, 1, r);
		}


#ifdef EAM2
		//Compute the electron density and the gradients
		if(type1 == type2){
#ifdef INTEL_SIMD
#pragma ivdep
#endif
#ifdef OMP
#pragma omp parallel for
#endif
			for (i=0; i<m; i++){
				//Clamp distance into the valid range of the potential table
				real r = MIN(r2[i+cellpairOffset], rhoEndCol1);
				r = MAX( 0.0, r - rhoBeginCol1);
				//Compute electron density rho and the gradient
				PAIR_INT_VEC(rho1[i], rho_grad1[i+cellpairOffset], rho_h_tab, n, 1, r);
			}
		} else {
#ifdef INTEL_SIMD
#pragma ivdep
#endif
#ifdef OMP
#pragma omp parallel for
#endif
			for (i=0; i<m; i++){
				int l = i+cellpairOffset;
				real r = MIN(r2[l], rhoEndCol1);
				r = MAX( 0.0, r-rhoBeginCol1);
				PAIR_INT_VEC(rho1[i], rho_grad1[l], rho_h_tab, col1, 1, r);
				real s = MIN(r2[l], rhoEndCol2);
				s = MAX( 0.0, s-rhoBeginCol2);
				PAIR_INT_VEC(rho2[i], rho_grad2[l], rho_h_tab, col2, 1, s);
			}
		}

		//Summing up rho
		for (i=0; i<m; i++){
			if (r2[i+cellpairOffset] < rhoEndCol1)
				EAM_RHO(cell_array+pair[i].p[0], pair[i].p[2]) += rho1[i];
			if (r2[i+cellpairOffset] < rhoEndCol2){
				if(type1==type2)
					EAM_RHO(cell_array+pair[i].p[1], pair[i].p[3]) += rho1[i];
				else EAM_RHO(cell_array+pair[i].p[1], pair[i].p[3]) += rho2[i];
			}
		}
#endif
		cellpairOffset+=m;	//Increase the offset in the

	} // pairs n

#ifdef EAM2

	//Sum up rho over buffer cells
	send_forces(add_rho,pack_rho,unpack_add_rho);

#ifdef OMP
#pragma omp parallel for
#endif
	/* compute embedding energy and its derivative */
	for (k=0; k<ncells; k++) {
		cell *p = CELLPTR(k);
		real pot, tmp, tr;
		const int n=p->n;
#ifdef INTEL_SIMD
#pragma ivdep
#endif
		for (i=0; i<n; i++) {
			int sorte = SORTE(p,i);
			real r = MAX( 0.0, EAM_RHO(p,i) - embed_pot.begin[sorte]);
			PAIR_INT_VEC(pot, EAM_DF(p,i), embed_pot, sorte, ntypes, r);
			POTENG(p,i) += pot;
		}
	}

	/* distribute derivative of embedding energy */
	send_cells(copy_dF,pack_dF,unpack_dF);
#endif

	//Reduce all results
	cellpairOffset = 0;
	for (n = 0; n < nPairs; n++) {
		const struct ppair* restrict pair = pairLists[n];

		const int m = pairsListLengths[n];

		const int type1 = n / ntypes;
		const int type2 = n % ntypes;
		const int col1 = n;
		const int col2 = type2 * ntypes + type1;

#ifdef EAM2
		real rhoCut = MAX(rho_h_tab.end[col1], rho_h_tab.end[col2]);
#endif

		for (i=0; i<m; i++) {
			vektor v, force;

			cell *p = cell_array+pair[i].p[0];
			cell *q = cell_array+pair[i].p[1];
			int n_i = pair[i].p[2];
			int n_j = pair[i].p[3];
			v.x = ORT(q, n_j, X) - ORT(p, n_i, X);
			v.y = ORT(q, n_j, Y) - ORT(p, n_i, Y);
			v.z = ORT(q, n_j, Z) - ORT(p, n_i, Z);

			real g = 0.;
			if (r2[i+cellpairOffset] < pair_pot.end[n]){
				g = grad[i+cellpairOffset];
				POTENG(p, n_i) += epot[i+cellpairOffset] * 0.5;
				POTENG(q, n_j) += epot[i+cellpairOffset] * 0.5;
			}

#ifdef EAM2
			if (r2[i+cellpairOffset] <= rhoCut) {
				real grad_df;
				if (type1==type2)
					g += 0.5 * (EAM_DF(p,n_i)+ EAM_DF(q,n_j)) * rho_grad1[i+cellpairOffset];
				else
					g += 0.5 * (EAM_DF(p,n_i) * rho_grad1[i+cellpairOffset] + EAM_DF(q,n_j) * rho_grad2[i+cellpairOffset]);
			}
#endif

			force.x = v.x * g;
			force.y = v.y * g;
			force.z = v.z * g;

			KRAFT(q, n_j,X) -= force.x;
			KRAFT(q, n_j,Y) -= force.y;
			KRAFT(q, n_j,Z) -= force.z;

			KRAFT(p, n_i,X) += force.x;
			KRAFT(p, n_i,Y) += force.y;
			KRAFT(p, n_i,Z) += force.z;

#ifdef P_AXIAL
			vir_xx -= v.x * force.x;
			vir_yy -= v.y * force.y;
			vir_zz -= v.z * force.z;
#else
			virial       -= SPROD(v,force);
#endif

#ifdef STRESS_TENS
			if (do_press_calc) {
				/* avoid double counting of the virial */
				force.x *= 0.5;
				force.y *= 0.5;
				force.z *= 0.5;

				PRESSTENS(p, n_i,xx) -= v.x * force.x;
				PRESSTENS(p, n_i,yy) -= v.y * force.y;
				PRESSTENS(p, n_i,xy) -= v.x * force.y;
				PRESSTENS(p, n_i,zz) -= v.z * force.z;
				PRESSTENS(p, n_i,yz) -= v.y * force.z;
				PRESSTENS(p, n_i,zx) -= v.z * force.x;
				PRESSTENS(q, n_j,zx) -= v.z * force.x;
				PRESSTENS(q, n_j,xx) -= v.x * force.x;
				PRESSTENS(q, n_j,yy) -= v.y * force.y;
				PRESSTENS(q, n_j,zz) -= v.z * force.z;
				PRESSTENS(q, n_j,xy) -= v.x * force.y;
				PRESSTENS(q, n_j,yz) -= v.y * force.z;
			}
#endif
		}
		cellpairOffset+=m;
	}//n pairs


	//Sum total potential energy
#ifdef OMP
#pragma omp parallel for reduction(+:tot_pot_energy)
#endif
	for (k=0; k<nallcells; k++) {
		cell *p = cell_array+k;
		int i;
		const int n = p->n;
		for (i=0; i<n; i++)
			tot_pot_energy += POTENG(p,i);
	}

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
		const int n = p->n;
		for (i = 0; i < n; i++) {
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

