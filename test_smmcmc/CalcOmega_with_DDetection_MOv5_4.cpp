#include "../include/micromegas.h"
#include"../include/micromegas_aux.h"
#include "lib/pmodel.h"
#include <string>

using namespace std;

/* ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
/* MAIN PROGRAM (by F.Staub, last change 02.01.2012)			     		    */
/* ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

int main(int argc, char** argv)
{  
		int err, i;
	   	char lspname[10], nlspname[10];
		double Omega=-1, Xf=-1;
		double w;
		double cut = 0.01;		// cut-off for channel output								
		int fast = 1;			/* 0 = best accuracy, 1 = "fast option" accuracy ~1% 	     */
 		double Beps = 1.E-5;  		/* Criteqrium for including co-annihilations (1 = no coann.) */
 		VZdecay=0; VWdecay=0; cleanDecayTable();
		ForceUG=1; 		
			err = sortOddParticles(lspname);	
			printMasses(stdout,1);				
	 		Omega = darkOmega(&Xf,fast,Beps,&err);
			printf("Xf=%.2e Omega h^2=%.2e\n",Xf,Omega);
//   			printChannels(Xf,cut,Beps,1,stdout);
			printf("\n");
			printChannels(Xf,cut,Beps,1,stdout);
			FILE *omega = fopen("omg.out","w");
			fprintf(omega,"%i %6.6lf # relic density \n",1,Omega);
			w = 1.;
			i = 0;
			while (w>cut) 
			{
			    fprintf(omega,"%i %6.6lf # %s %s -> %s %s\n",100+i,omegaCh[i].weight,omegaCh[i].prtcl[0],omegaCh[i].prtcl[1],omegaCh[i].prtcl[2],omegaCh[i].prtcl[3]);
			    i++;
			    w = omegaCh[i].weight;
			}
				FILE *channels = fopen("channels.out","w");
			w = 1.;
			i = 0;
			while (w>cut) 
			{
			fprintf(channels,"%li %li %li %li %6.6lf # %s %s -> %s %s\n",pNum(omegaCh[i].prtcl[0]),pNum(omegaCh[i].prtcl[1]),pNum(omegaCh[i].prtcl[2]),pNum(omegaCh[i].prtcl[3]),omegaCh[i].weight,omegaCh[i].prtcl[0],omegaCh[i].prtcl[1],omegaCh[i].prtcl[2],omegaCh[i].prtcl[3]);
			    i++;
			    w = omegaCh[i].weight;
			}


//-------------------------------------------//
//      Direct detection calculation         //
//-------------------------------------------//

char cdmName[10], CDM1[10]; // Asegúrate que esto esté al inicio del main
err = sortOddParticles(cdmName);
strcpy(CDM1, cdmName);

double pA0[2], pA5[2], nA0[2], nA5[2];
double Nmass = 0.939; // nucleon mass in GeV
double SCcoeff;

double dNdE[300];
double nEvents;

// Amplitudes de dispersión
nucleonAmplitudes(CDM1, pA0, pA5, nA0, nA5);

// Coeficiente para convertir a secciones eficaces en pb
SCcoeff = 4 / M_PI * 3.8937966E8 * pow(Nmass * Mcdm / (Nmass + Mcdm), 2.);

// Secciones eficaces [pb]
double sigSIP = SCcoeff * pA0[0] * pA0[0];   // spin-independent proton
double sigSIN = SCcoeff * nA0[0] * nA0[0];   // spin-independent neutron
double sigSDP = 3 * SCcoeff * pA5[0] * pA5[0]; // spin-dependent proton
double sigSDN = 3 * SCcoeff * nA5[0] * nA5[0]; // spin-dependent neutron

// Conversión pb -> cm²
double sigSIP_cm2 = sigSIP * 1.0e-36;
double sigSIN_cm2 = sigSIN * 1.0e-36;
double sigSDP_cm2 = sigSDP * 1.0e-36;
double sigSDN_cm2 = sigSDN * 1.0e-36;

// Imprimir resultados
printf("\n==== Direct Detection: CDM-Nucleon Cross Sections ====\n");
printf(" proton  SI %.3E    SD %.3E \n", sigSIP_cm2, sigSDP_cm2);
printf(" neutron SI %.3E    SD %.3E \n", sigSIN_cm2, sigSDN_cm2);



printf("\n==== Indirect Detection =======\n");

double sigmaV;
double SpA[NZ], SpE[NZ], SpP[NZ];
double FluxA[NZ], FluxE[NZ], FluxP[NZ];
double *SpNe = NULL, *SpNm = NULL, *SpNl = NULL;

sigmaV = calcSpectrum(1 + 2 + 4, SpA, SpE, SpP, SpNe, SpNm, SpNl, &err);

// Mostrar el valor total de la sección eficaz de aniquilación
printf("⟨σv⟩ = %.3E cm³/s\n", sigmaV);




       fclose(channels);
       fclose(omega);

  	return 0;
}

