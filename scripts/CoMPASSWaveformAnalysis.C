#include "TArray.h"
#include "TGraph.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TF1.h"

UShort_t QuickQIntegration(TArrayS*& waveform);
Float_t Gain = 4.06E+6; ///< Previously defined gain
Float_t e = 1.602E-19; ///< elementary charge
UShort_t Vbin = 2000 / pow(2,14); ///< Voltage binning in mV
UShort_t Tbin = 2; ///< Timing binning in ns
UShort_t preGate = 36; ///< Set the pre-gate 
UShort_t Gate    = 100; // gate length in ns

void CoMPASSWaveformAnalysis(){
  std::string sInFile = "/media/robertkralik/LaCie/TestingPMTs/DAQ/ComparisonLonger/UNFILTERED/compass_ComparisonLonger.root";
  TFile fIn(sInFile.c_str(),"read");

  TTree* tree = (TTree*)fIn.Get("Data");
  
  //Create the variables that will store the information
  UShort_t Energy, EnergyShort;
  TArrayS* Samples = new TArrayS;
  //Asign variables to TBranches
  tree->SetBranchAddress("Energy", &Energy);
  tree->SetBranchAddress("EnergyShort", &EnergyShort);
  tree->SetBranchAddress("Samples", &Samples);
  //int NEntries = 10;
  int NEntries = tree->GetEntries();

  TH1F hCharge("hCharge",";Charge [pC];Counts",70,0,70);
  TH1F hEnergy("hEnergy",";Energy [ADC Counts];Counts",170,0,170);
  TH1F hEnergyShort("hEnergyShort",
                    ";Energy (short) [ADC Counts];Counts",170,0,170);
  TH2F hChargeEnergy("hChargeEnergy",
                     ";Charge [pC];Energy [ADC Counts]",70,0,70,170,0,170);
  TH2F hChargeEnergyShort("hChargeEnergyShort",
                          ";Charge [pC];Energy (short) [ADC Counts]",
                          70,0,70,170,0,170);
  TH2F hEnergyEnergyShort("hEnergyEnergyShort",
                          ";Energy [ADC Counts];Energy (short) [ADC Counts]",
                          170,0,170,170,0,170);

  for(int iEntry = 0; iEntry < NEntries; iEntry++){
    tree->GetEntry(iEntry);
    UShort_t ADCCount = QuickQIntegration(Samples);
    Float_t  Charge   = ADCCount / (1.024*50); // charge in pC
    Float_t  NPE      = Charge*1E-12 / (Gain*e); // Number of PE

    hCharge.Fill(Charge);
    hEnergy.Fill(Energy);
    hEnergyShort.Fill(EnergyShort);
    hChargeEnergy.Fill(Charge,Energy);
    hChargeEnergyShort.Fill(Charge,EnergyShort);
    hEnergyEnergyShort.Fill(Energy,EnergyShort);
    /*
    std::cout << "Energy: " << Energy
              << ", EnergyShort: " << EnergyShort
              << ", Total ADC count: " << ADCCount
              << ", Charge: " << Charge
              << ", NPE: " << NPE
              << ", Samples: ";
    for(int iSample = 0; iSample < Samples->GetSize(); iSample++){
      std::cout << Samples->GetAt(iSample) << ",";
    }
    std::cout << std::endl;
    */
  }

  TF1 *f1 = new TF1("logNormal","[0]*ROOT::Math::lognormal_pdf(x,[1],[2])",0,150);
  Double_t p[3];
  p[0] = hEnergy.GetEntries()*hEnergy.GetXaxis()->GetBinWidth(1);
  double prob[] = {0.5}; 
  double q[1]; 
  hEnergy.GetQuantiles(1,q,prob);
  double median = q[0];
  // find mode of histogram 
  double  mode = hEnergy.GetBinCenter( hEnergy.GetMaximumBin());
  p[1] = std::log(median);
  p[2] = std::sqrt( std::log(median/mode) );
  f1->SetParameters(p); 
  f1->SetParName(0,"A");
  f1->SetParName(1,"m");
  f1->SetParName(2,"s");
  hEnergy.Fit(f1,"V");

  TFile fOut("CoMPASSHistogramsLonger.root","recreate");
  fOut.cd();
  hCharge.Write();
  hEnergy.Write();
  hEnergyShort.Write();
  hChargeEnergy.Write();
  hChargeEnergyShort.Write();
  hEnergyEnergyShort.Write();
  fOut.Close();
  fIn.Close();
}

void PlotWaveform(TArrayS*& waveform){
  TGraph graph("gr");
  for(UShort_t iSample = 0; iSample < waveform->GetSize(); iSample++){
    UShort_t time = iSample*Tbin;
    gr.AddPoint(time,waveform->GetAt(iSample));
  }
  gr.Draw()
}

UShort_t QuickQIntegration(TArrayS*& waveform){
  UShort_t Baseline = 940; // hardcode baseline for now
  UShort_t GateStart = 36; // doesn't need to be precise - start of gate
  UShort_t Gate      = 100; // gate length in ns
  UShort_t Charge = 0;
  for(UShort_t iSample = GateStart; iSample < GateStart+Gate; iSample++){
    Charge += Baseline - waveform->GetAt(iSample);
  }
  return Charge;
}