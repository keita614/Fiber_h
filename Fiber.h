#ifndef ___FIBER_H___
#define ___FIBER_H___

/*!
 * このクラスは、ファイバー内で光が一様等方に伝播することを前提として、
 * そのTrapping Efficiencyのを計算するために作成した。
 *
 * December 2024
 * T. Kobayashi, I. Komae, and Y. Tsunesada
 * Osaka Metropolitan University
 * 
 * 2025年10月以降　湯淺　圭太も参画
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cmath>
#include <unistd.h>
#include <vector>
#include <sys/stat.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <omp.h>
#include <gsl/gsl_errno.h>
#include "gsl/gsl_integration.h"
#include "gsl/gsl_deriv.h"

using namespace std;

class Fiber {
public:
  // ===========================================================================================
  //  1. コンストラクタの設定
  // ===========================================================================================
  /*!
   * Constructor for a fiber
   * @param n0 The refractive index of the fiber core
   * @param n1 The refractive index of the inner cladding
   * @param d1 The radius of the fiber core in [mm]
   * @param L The length of the fiber in [mm]
   * @param Latt The atteunuation length of the fiber in [mm]
   * @param Labs The absorption length of the fiber in [mm]
   */
  Fiber(double n0, double n1, double d1, double L, double Latt, double Labs) {
    _n0 = n0;
    _n1 = n1;
    _n2 = -9.99;
    _d1 = d1;
    _L = L;
    _Latt = Latt;
    _Labs = Labs;       
  }
  /*!
   * Constructor for a fiber
   * @param n0 The refractive index of the fiber core
   * @param n1 The refractive index of the inner cladding
   * @param n2 The refractive index of the outer cladding
   * @param d1 The radius of the fiber core in [mm]
   * @param L The length of the fiber in [mm]
   * @param Latt The atteunuation length of the fiber in [mm]
   * @param Labs The absorption length of the fiber in[mm]
   */
  Fiber(double n0, double n1, double n2, double d1, double L, double Latt, double Labs) {
    _n0 = n0;
    _n1 = n1;
    _n2 = n2;
    _d1 = d1;
    _L = L;
    _Latt = Latt;
    _Labs = Labs;
  }  

  // ===========================================================================================
  //  2. メンバ変数のセッターとゲッター
  // ===========================================================================================

  /*! Returns the refactive index of the fiber core */
  double GetN0() const { return _n0; }
  /*! Returns the refactive index of the inner cladding */  
  double GetN1() const { return _n1; }
  /*! Returns the refactive index of the outer cladding */  
  double GetN2() const { return _n2; }
  /*! Returns the radius of the fiber core in [mm] */
  double GetD1() const { return _d1; }
  /*! Returns the outer radius of the inner cladding in [mm] */  
  double GetD2() const { return _d2; }
  /*! Returns the fiber length in [mm] */
  double GetL() const { return _L; }
  /*! Returns the atteunuation length of the fiber in [mm] */
  double GetLatt() const { return _Latt; }
  /*! Returns the absorption length of the fiber in [mm]*/
  double GetLabs() const { return _Labs; }
  double GetTheta() const { return _theta; }
  void SetTheta(double theta) { _theta = theta; }
  double GetPhi() const { return _phi; }
  void SetPhi(double phi) { _phi = phi; }
  double GetPsi() const {return _psi; }
  void SetPsi(double psi){ _psi = psi;}
  double GetMu() const { return _mu; }
  void SetMu(double mu) { _mu = mu; }      
  double GetA() const { return _a; }
  void SetA(double a) { _a = a; }
  double GetZ() const { return _z; }
  void SetZ(double z) { _z = z; }
  double GetP() const { return _P; }
  double GetPerr() const { return _Perr; }  
  double GetPatt() const { return _Patt; }
  double GetPatterr() const { return _Patterr; } 
  double GetLightSpeed() const { return 299.792458; } //光速 [mm/ns] 

  // ===========================================================================================
  //  3. 計算が行われる関数たち
  // ===========================================================================================

  // 仕様書3章 //

  /**
   * 仕様書式(9)の計算本体
   * dN/daについて、psi=0つまりファイバーに垂直に入射した場合の身を考える
   */
  void Check_a_dist_vertical(const char *fileName, double aStep = 0.005){
    FILE *fp = fopen(fileName, "w");
    double a = 0.0; 
    double psi = 0.0;
    double dA = aStep;
    while(a < GetD1()){
      double Adist = a_dist_with_abs(a, psi);
      fprintf(fp, "%f %f\n", a, Adist);
      printf("Finished a %f mm\n", a);
      a += dA;
    }
    fclose(fp);
  }

  /**
   * 仕様書式(10)の計算本体
   * dN/daをψで平均化したものの分布を確認することができる
   * @param psi はファイバー軸に垂直な軸からの角度である
   */
  void Check_a_dist(const char *fileName, double aStep = 0.005){
    FILE *fp = fopen(fileName, "w");
    double a = 0.0; 
    double dA = aStep;
    while(a < GetD1()){
      double Adist = Calc_a_dist_average_over_psi(a);
      fprintf(fp, "%f %f\n", a, Adist);
      printf("Finished a %f mm\n", a);
      a += dA;
    }
    fclose(fp);
  }

  // 仕様書4章, 5章 //

  /**
   * 仕様書式(13), (21)の計算本体
   * Trapping Efficiencyのa分布について、atenuationの効果があるものとないものをそれぞれ計算する
   */
  void Calc_Pa_Patta(const char *fileName, double astep = 0.005) {
    FILE *fp = fopen(fileName, "w");
    Calc_P_average_over_a();
    Calc_P_average_over_za();
    fprintf(fp, "# a P(a) Patt(a)\n");
    fprintf(fp, "# n0 = %3.2f, n1 = %3.2f, n2 = %3.2f\n", _n0, _n1, _n2);
    fprintf(fp, "# d = %f mm, L = %f mm, Latt = %f mm\n", _d1, _L, _Latt);
    fprintf(fp, "# P = %e +/- %e, Patt = %e */- %e\n", _P, _Perr, _Patt, _Patterr);
    fprintf(fp, "%e  %e\n", _P, _Patt);
    printf("P = %e +/- %e, Patt = %e +/- %e\n", _P, _Perr, _Patt, _Patterr);

    double a = 0.0;
    while (a < _d1) {
      double dNdA = Calc_a_dist_average_over_psi(a);      
      double Paresult = Calc_Pa(a);      
      double Pattaresult = Calc_Pa_average_over_z(a);
      fprintf(fp, "%e %e %e\n", a, Paresult*dNdA, Pattaresult*dNdA);
      a += 0.005;
    }
    a = _d1;
    double dNdA = Calc_a_dist_average_over_psi(a);    
    double Paresult = Calc_Pa(a);    
    double Pattaresult = Calc_Pa_average_over_z(a);
    fprintf(fp, "%e %e %e\n", a, Paresult*dNdA, Pattaresult*dNdA);    
    fclose(fp);
  }

  // 6.1 角度分布 //

  void Calc_dPdTheta(const char *fileName, double thetaStep = 1.0) {
    double theta = 0.0;
    double dTheta = thetaStep*M_PI/180;
    vector<double> v;
    double sum = 0.0;
    while (theta < M_PI/2) {
      double P = dPdTheta(theta);
      v.push_back(P);      
      sum += P*dTheta;
      theta += dTheta;
    }    
    FILE *fp = fopen(fileName, "w");
    fprintf(fp, "theta dPdTheta\n");
    theta = 0.0;
    for (size_t i = 0; i < v.size(); i++) {
      fprintf(fp, "%f %f\n", theta*180/M_PI, v[i]/sum);
      theta += dTheta;
    }
    fclose(fp);
  }

  void Calc_dPdTheta_secondtheta(const char *fileName, double SecthetaStep = 0.01) {
    double SecTheta = 1;
    double dSecTheta = SecthetaStep;
    double c = 1/SecTheta;
    double s = sqrt(1-c*c);
    double theta = acos(c);
    vector<double> v;
    double sum = 0.0;
    double P = 0.0;
    while (SecTheta < 10) {
      s = sqrt(1-c*c);
      c = 1/SecTheta;
      theta = acos(c);
      if (s > 1e-9) { // 0.0 と厳密に比較する代わりに、非常に小さい数より大きいかを見る
      P = c*c*dPdTheta(theta);}
      v.push_back(P);      
      sum += P*dSecTheta;
      SecTheta += dSecTheta;
    }    
    FILE *fp = fopen(fileName, "w");
    SecTheta = 1;
    for (size_t i = 0; i < v.size(); i++) {
      double x_val = SecTheta; 
      double y_val = 0.0; // デフォルトは0
      if (fabs(sum) > 1e-9) { // sumが0(またはほぼ0)でないことを確認
          y_val = v[i] / sum;
      }
      fprintf(fp, "%f %f\n", x_val, y_val);
      SecTheta += dSecTheta;
    }
    fclose(fp);
  }

  void Calc_dPda(const char *fileName, double aStep = 0.01) {
    double a = 0;
    vector<double> v;
    double sum = 0.0;
    while (a <= _d1) {
      double p = dPda(a);
      v.push_back(p);
      sum += p*aStep;
      a += aStep;
    }
    FILE *fp = fopen(fileName, "w");
    fprintf(fp, "a dPda\n");
    a = 0;
    for (size_t i = 0; i < v.size(); i++) {
      fprintf(fp, "%f %f\n", a, v[i]/sum);
      a += aStep;
    }
    fclose(fp);
  }
  
  //式(31)の計算をステップを刻んで出力
 void Calc_dPdt(const char *fileName, double thetaStep = 10, double aStep = 0.05 , double zStep = 100){
    FILE *fp = fopen(fileName, "w");
    double theta = 0.0;
    double dTheta = thetaStep*M_PI/180;
    double a = 0.0;
    double dA = aStep;
    double z = 0.0;
    double dZ = zStep;
    fprintf(fp, "theta a z t P\n");
    while (theta < M_PI/2) {
      a = 0.0;
      while (a < GetD1()) {
        z = 0.0;
        while (z < GetL()) {
          double P = dPdt(theta, a, z);
          double t = Transit_time(theta, z);
          fprintf(fp, "%f %f %f %f %f\n", theta*180/M_PI, a, z, t, P);
          z += dZ;
        }
        a += dA;
      }
      theta += dTheta;
    }
    fclose(fp);
  }  

  //式(32)の計算をステップを刻んで出力
  void Calc_dPdt_average_over_theta(const char *fileName, double thetaStep = 1.0, double aStep = 0.05 , double zStep = 10){
    FILE *fp = fopen(fileName, "w");
    double theta = 0.0;
    double dTheta = thetaStep*M_PI/180;
    double a = 0.0;
    double dA = aStep;
    double z = 0.0;
    double dZ = zStep;
    double weight = 0.0;
    fprintf(fp, "a z t P\n");

    while (z < GetL()) {
      a = 0.0;
      while (a < GetD1()) {
        double sum_P = 0.0;
        double sum_t = 0.0;
        double sum_weight = 0.0;
        theta = 0.0;
        while (theta < M_PI/2) {
          double P = dPdt(theta, a, z);
          double t = Transit_time(theta, z);
          if (theta > 0.0) {weight = sin(theta) * dTheta;}
          else { weight = 0.0;};
          double P_weight = P * weight;
          double t_weight = t * weight;
          sum_P += P_weight;
          sum_t += t_weight;
          sum_weight += weight;

          theta += dTheta;
        }
        double avg_P = (sum_weight > 0) ? (sum_P / sum_weight) : 0.0;
        double avg_t = (sum_weight > 0) ? (sum_t / sum_weight) : 0.0;
        fprintf(fp, "%f %f %f %e\n", a, z, avg_t, avg_P);
        a += dA;
      }
      printf("Finished z = %f mm\n", z);
      z += dZ;
    }
    fclose(fp);
  }

  // Absorption Length の効果を考慮に入れたもの
  void Calc_dPdt_with_a_dist(const char *fileName, double thetaStep = 1.0, double aStep = 0.05 , double zStep = 10){
    FILE *fp = fopen(fileName, "w");
    double theta = 0.0;
    double dTheta = thetaStep*M_PI/180;
    double a = 0.0;
    double dA = aStep;
    double z = 0.0;
    double dZ = zStep;
    double weight = 0.0;
    fprintf(fp, "a, z, t, dPdt\n");

    while (z < GetL()) {
      a = 0.0;
      while (a < GetD1()) {
        double sum_P = 0.0;
        double sum_t = 0.0;
        double sum_weight = 0.0;
        double weight_a = Get_a_initial_distribution(a);
        theta = 0.0;
        while (theta < M_PI/2) {
          double P = dPdt(theta, a, z);
          double t = Transit_time(theta, z);
          if (theta > 0.0) {weight = sin(theta) * dTheta;}
          else { weight = 0.0;};
          double P_weight = P * weight * weight_a;
          double t_weight = t * weight;
          sum_P += P_weight;
          sum_t += t_weight;
          sum_weight += weight;

          theta += dTheta;
        }
        double avg_P = (sum_weight > 0) ? (sum_P / sum_weight) : 0.0;
        double avg_t = (sum_weight > 0) ? (sum_t / sum_weight) : 0.0;
        fprintf(fp, "%f %f %f %e\n", a, z, avg_t, avg_P);
        a += dA;
      }
      printf("Finished z = %f mm\n", z);
      z += dZ;
    }
    fclose(fp);
  }

  // Absorption Length の効果を考慮に入れ、aで平均化したもの
  void Calc_dPdt_a_with_a_dist(const char *fileName, double thetaStep = 1, double zStep = 1){
    // Structure to hold results to avoid concurrent file writing
    struct DataPoint {
      double z;
      double t;
      double dPdt;
    };
    
    // Calculate number of steps
    int numSteps = 0;
    for(double z = 0.0; z < GetL(); z += zStep) numSteps++;
    
    vector<DataPoint> results(numSteps);
    double dTheta = thetaStep * M_PI / 180.0;

    PrepareTable(); // Ensure the table is loaded before parallel region
    
    printf("Starting parallel calculation with OpenMP...\n");

    // OpenMP Parallel Loop
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < numSteps; i++) {
      double z = i * zStep;
      
      // COPY the fiber object for this thread to ensure thread-safety
      Fiber localFiber = *this; 

      double sum_P = 0.0;
      double sum_weight = 0.0;
      double theta = 0.0;
      double t_final = 0.0;

      while(theta < M_PI/2){
        // Use localFiber for calculations
        double P = localFiber.dPdt_a(z, theta);
        double t = localFiber.Transit_time(theta, z);
        
        double weight;
        if (theta > 0.0) {weight = sin(theta) * dTheta;}
        else { weight = 0.0;};
        
        double P_weight = P * weight ;
        sum_P += P_weight;
        sum_weight += weight;
        
        theta += dTheta;
        t_final = t; // Store t (as in original code logic, though implies averaging might be better)
      }
      // Normalize
      double avg_P = (sum_weight > 0) ? (sum_P / sum_weight / localFiber.GetL()) : 0.0;
      
      // Store result
      results[i].z = z;
      results[i].t = t_final;
      results[i].dPdt = avg_P;
    }

    // Sequential File Writing
    FILE *fp = fopen(fileName, "w");
    fprintf(fp, "z t dPdt\n");
    for(const auto& pt : results) {
        fprintf(fp, "%f %f %e\n", pt.z, pt.t, pt.dPdt);
    }
    fclose(fp);
    printf("Calculation finished and saved to %s\n", fileName);
  }

  // zは固定し、Absorption Length の効果を考慮に入れ、aで平均化したもの
  void Calc_dPdt_a_with_a_dist_solidz(const char *fileName, double thetaStep = 1){
    FILE *fp = fopen(fileName, "w");
    double theta = 0.0;
    double dTheta = thetaStep*M_PI/180;
    double z = 0;
    double t;
    double weight = 0.0;

    double P_weight;
    theta = 0.0;
    fprintf(fp, "t dPdt\n");
    while(theta < M_PI/2){
      double P = dPdt_a(z, theta);
      t = Transit_time(theta, z);
      if (theta > 0.0) {weight = sin(theta) * dTheta;}
        else { weight = 0.0;};
      P_weight = P * weight ;

      fprintf(fp, "%f %e\n", t, P_weight);
      printf("Calculating... Finished theta = %f \n",theta);
      fflush(stdout);
      theta += dTheta;
    }

    fclose(fp);
  }

  void Calc_escape_angle(const char *fileName, double thetaStep = 1.0){
    double theta = 0.0;
    double dTheta = thetaStep*M_PI/180;
    vector<double> v;
    double sum = 0.0;
    while (theta < M_PI/2) {
      double n_ice = 1.309;
      double arg = GetN0()/n_ice * sin(theta);
      if (arg > 1.0) {
          break; 
      }
      double P = dPdTheta_escape(theta);
      v.push_back(P);      
      sum += P*dTheta;
      theta += dTheta;
    }    
    FILE *fp = fopen(fileName, "w");
    fprintf(fp, "theta dPdTheta\n");
    theta = 0.0;
    for (size_t i = 0; i < v.size(); i++) {
      double n_ice = 1.309;
      double arg = GetN0()/n_ice * sin(theta);
        if (arg > 1.0) {
            break; 
        }
      double theta_escape = asin(arg);
      fprintf(fp, "%f %f\n", theta_escape*180/M_PI, v[i]/sum);
      theta += dTheta;
    }
    fclose(fp);
  }

private:
  // ===========================================================================================
  //  1. メンバ変数 (Member Variables)
  //     - クラス全体で使うデータはここにまとめる
  // ===========================================================================================

  /*! Fiber attributes */
  double _n0; /*!< Refractive index of the fiber core */
  double _n1; /*!< Refractive index of the inner cladding */
  double _n2; /*!< Refractive index of the outer cladding */
  double _d1; /*!< Radius of the fiber core in [mm] */
  double _d2; /*!< Outer radius of the inner cladding in [mm] */
  double _L;  /*!< Fiber length in [mm] */
  double _Latt; /*!< Fiber attenuation length in [mm] */
  double _Labs; /*!< Fiber absorption length in [mm] */
  /*! Temporal variables */
  double _mu;   /*!< Cosine(Theta) */
  double _theta;
  double _phi;  /*!< Azimuthal angle in radians */
  double _psi;  /*!< Azimuthal angle with respect to the cross-section of the fiber*/
  double _a;    /*!< Distance from the fiber axis in [mm] */
  double _z;    /*!< Distance from an edge of the fiber along the axis in [mm] */
  double _result, _err;
  double _P, _Perr;
  double _Patt, _Patterr;
  /*データをメモリに一時保持するための変数*/
  vector<double> table_a; //aの値
  vector<double> table_val; //計算された初期位置分布の値
  bool is_table_loaded = false; //読み込み済みフラグ
  const string table_filename = "Initial_a_distribution.txt"; //保存ファイル名

  // ===========================================================================================
  //  2. gslを用いた積分用のstatic関数
  //      見なくていい、理解したいなら頑張れ
  // ===========================================================================================

  // 仕様書3章　ファイバー内で発光する初期位置a //

  /**
   * 仕様書式(10)の被積分項
   * aの初期位置の分布dN/daをについて、ファイバーへの入射角ψで平均化したもの
   */
  static double f_integral_a_dist_over_psi(double psi, void *data){
    Fiber *fiber = (Fiber *) data;
    double a = fiber->GetA();
    return fiber->a_dist_with_abs(a, psi);
  }

  // 仕様書4章 コア軸からの距離a の関数としてのTrapping Eﬃciency　//

  /**
   * 仕様書式(11)の被積分項
   * 任意のaに対するTrapping Efficiency P(a)の被積分項を取得
   */
  static double f_integral_solidangle(double phi, void *data) {
    Fiber *fiber = (Fiber *) data;
    double a = fiber->GetA();  
    return fiber->Calc_cosThetaMax(phi, a);
  }

  /**
   * 仕様書式(13)の被積分項
   * Trapping Efficiency P(a)をaで積分するための被積分項
   */
  static double f_integral_average_over_a(double a, void *data) {
    Fiber *fiber = (Fiber *) data;
    double Pa = fiber->Calc_Pa(a);
    double dNdA = fiber->Calc_a_dist_average_over_psi(a);
    return dNdA*Pa;
  }

  //　仕様書5章　ファイバー端からの距離z でのTrapping Eﬃciency　//

  /**
   * 仕様書式(19)の被積分項
   * 光がファイバー内で減衰する効果を表す項
   */
  static double f_integral_mu_att(double mu, void *data) {
    Fiber *fiber = (Fiber *) data;
    double L = fiber->GetL();
    double Latt = fiber->GetLatt();  
    double z = fiber->GetZ();
    if (fabs(mu) < 1e-6) return 0;
    return exp(-(L-z)/mu/Latt);
  }

  /**
   * 仕様書式(19)の積分本体
   * 任意のzでのTrapping Effisciency P(a,z)のμ積分の項
   * @param z は検出側とは逆のファイバー端からの距離
   */
  static double f_integral_phi_att(double phi, void *data) {
    Fiber *fiber = (Fiber *) data;
    gsl_function F;
    F.function = f_integral_mu_att;
    F.params = data;
    double a = fiber->GetA();
    double mumin = fiber->Calc_cosThetaMax(phi, a);
    double result, err;
    gsl_integration_workspace *local_w = gsl_integration_workspace_alloc(1000);
    gsl_integration_qag(&F, mumin, 1, 0, 1e-6, 1000, GSL_INTEG_GAUSS31, local_w, &result, &err);
    gsl_integration_workspace_free(local_w); 
    return result;
  }
  
  /**
   * 仕様書式(20)の被積分項
   * 任意のzでのTrapping Effisciency P(a,z)の結果をzで平均化するための被積分項
   */
  static double f_integral_average_over_z(double z, void *data) {
    Fiber *fiber = (Fiber *) data;
    double a = fiber->GetA();
    double Paz = fiber->Calc_Paz(a, z);
    return Paz;
  }
  
  /**
   * 仕様書式(21)の被積分項
   * zで平均化されたTrapping Effisciency P(a,z)をaで平均化するための被積分項
   */
  static double f_integral_average_over_za(double a, void *data) {
    Fiber *fiber = (Fiber *) data;
    double Pa = fiber->Calc_Pa_average_over_z(a);
    double dNdA = fiber->Calc_a_dist_average_over_psi(a);
    return dNdA*Pa;
  }

  // 仕様書6章1節 角度分布 //

  /**
   * 仕様書式(22)の積分範囲の決定
   * ∮_Φ(θ)dΦで積分できるなら1をできないなら0を返し擬似的に積分範囲を決定
   */
  static double check_refrection(double phi, void *data) {
    Fiber *fiber = (Fiber *) data;
    double sinTheta = sin(fiber->GetTheta());
    double sinPhi = sin(phi);
    double a = fiber->GetA();
    double d = fiber->GetD1();
    double cosPsi = sinTheta*sqrt(1.0 - a*a/d/d*sinPhi*sinPhi); //常定pdfの(7)式の計算
    double sinPsi = sqrt(1.0 - cosPsi*cosPsi); //前式よりsinPsiを計算
    double nn;
    //常定pdfの(16)式の計算を場合分けで行なっている
    if (fiber->GetN2() < 0) { // single cladding
      nn = fiber->GetN1()/fiber->GetN0();
    } else {
      nn = fiber->GetN2()/fiber->GetN0(); // double cladding     
    }
    if (sinPsi < nn) return 0.0;
    //    else return sinTheta;
    else return 1.0;
  }

  /**
   * 仕様書式(23)の被積分項
   * Trapping Efficiencyの角度分布dP/dθをファイバー全体で平均化する計算の被積分項
   */
  static double f_integral_theta_dist_a(double a, void *data) {
    Fiber *fiber = (Fiber *) data;
    double theta = fiber->GetTheta();
    double dPdTheta = fiber->dPdTheta(a, theta) * fiber->Get_a_initial_distribution(a);
    return dPdTheta;
  }

  // 仕様書6章3節 //

  /**
   * 仕様書式(31)の被積分項
   * Trapping Efficiencyの時間分布dP/dtをθで平均化する計算の被積分項
   */
  static double f_integral_a_dist(double a, void *data){
    Fiber *fiber = (Fiber *) data;
    double theta = fiber->GetTheta();
    double z = fiber->GetZ();
    double dPdt = fiber->dPdt(theta, a, z)*fiber->Get_a_initial_distribution(a);
    return dPdt;
  }

  // 仕様書6章5節
  /**
   * 仕様書式()　＜ーちゃんと式かけ
   * ファイバー端から脱出する光の分布をTrapping Efficiencyのθ分布から計算し、それをaで平均化する計算の被積分項
   */
  static double f_integral_escape_dist(double a, void *data){
    Fiber *fiber = (Fiber *) data;
    double theta = fiber->GetTheta();
    double dPdTheta = fiber->dPdTheta(a, theta);
    double weight = fiber->Escape_angle_distribution(theta, a);
    return weight*dPdTheta;
  }

  // 未記載 //
  /*
   * Komae's (18)
   */
  //ファイバーを出ていく光子が中心軸からどれだけ離れているかの計算
  static double f_aprime(double a, void *data) {
    Fiber *fiber = (Fiber *) data;
    double theta = fiber->GetTheta();
    double phi = fiber->GetPhi();
    double z = fiber->GetZ();
    double L = fiber->GetL();
    double d = fiber->GetD1();
    double sinphi = sin(phi);
    double cosphi = cos(phi);
    double b = sqrt(d*d - a*a*sinphi*sinphi); //光子の反射点から次の反射点までの水平方向の距離の半分
    double A = (L - z)*tan(theta); //光子の水平方向の総移動距離
    double ltotal = A + a*cosphi + b; //光子の水平方向の総移動距離
    double l = 2.0*b;
    int n = (int) (ltotal/l); //反射回数。少数切り捨て
    double B = 2.0*n*b; //反射しながら進んだ合計の距離
    double AB = A - B; //最後の反射が終わった後の、ファイバー終端に達するまでの移動距離
    double aprime2 = a*a + 2.0*a*cosphi*AB + AB*AB; //ファイバーを出ていく光子が中心軸からどれだけ離れているか
    return sqrt(aprime2);
  }

  // ===========================================================================================
  //  3. gsl以外のprivateな関数
  //      ここも見なくていいが、新しい関数を作成したいのなら理解したほうがいい
  // ===========================================================================================

  // 仕様書3章 //

  /**
   * 仕様書式(9)の計算
   * absorptionの効果を入れた発光点の初期位置の分布の計算
   */
  inline double a_dist_with_abs(double a, double psi){
    double r = GetD1();
    double l_abs = GetLabs();
    double s = sin(psi);
    double c = cos(psi);
    double root_arg = a*a - r*r*s*s;
    if (root_arg <= 0){return 0.0;}
    double root = sqrt(root_arg);
    double ch = cosh(root/l_abs);
    double Adist_val = 2.0/l_abs * exp(-r*c/l_abs) * ch * a / root;
    return Adist_val;
  }

  /**
   * 仕様書式(10)の積分本体
   * aの初期位置の分布dN/daをについて、ファイバーへの入射角ψで平均化したもの
   */
  double Calc_a_dist_average_over_psi(double a){
    if(isnan(_Labs)){return 1.0;}
    else{
      SetA(a);
      double upper_limit;
      double r = GetD1();
      if (a >= r) {
          upper_limit = M_PI / 2.0; // aがrより大きければ常時正なので90度まで積分
      } else {
          upper_limit = asin(a / r); // 限界角度まで積分
      }
      gsl_function F;
      F.function = f_integral_a_dist_over_psi;
      F.params = this;
      double result, err;
      double epsabs = 1e-10; // 絶対誤差の許容値
      double epsrel = 1e-5;  // 相対誤差の許容値
      gsl_integration_workspace *local_w = gsl_integration_workspace_alloc(1000);
      gsl_integration_qag(&F, 0, M_PI/2, epsabs, epsrel, 1000, GSL_INTEG_GAUSS41, local_w, &result, &err);
      gsl_integration_workspace_free(local_w);
      return result;
    }
  }

  // Initial_a_distribution.txtの準備
  void PrepareTable() {
    if (is_table_loaded) return;

    // ファイル存在確認
    struct stat buffer;
    bool file_exists = (stat(table_filename.c_str(), &buffer) == 0); //stat()でファイルの存在を確認 → .c_str()でstring 文字列を、C言語形式の文字列（char*）に変換

    // ファイルがない場合のみ作成 (重い処理はここだけ)
    if (!file_exists) {
      cout << "Generating table file: " << table_filename << " ..." << endl; //printfみたいなもん。左から順番に出力され、endlで改行
      ofstream outfile(table_filename);
      outfile << "# a  Initial_a_distribution_Mean" << endl;

      double a = 0.0;
      double limit = GetD1();
      double step = 0.00001; // 精度が必要なら細かくする

      while (a < limit) {
          double val = Calc_a_dist_average_over_psi(a); // 重い計算を実行
          outfile << a << " " << val << endl;
          a += step;
      }
      outfile.close();
      cout << "Generation completed." << endl;
    }

    // ファイル読み込み (メモリへの展開)
    ifstream infile(table_filename);
    if (!infile) {
      cerr << "Error: Cannot open table file." << endl;
      return;
    }

    double t_a, t_val;
    string line;
    table_a.clear();
    table_val.clear();

    while (getline(infile, line)) { //ファイルから「1行まるごと」文字列として読み込み
      if (line.empty() || line[0] == '#') continue;
      stringstream ss(line); //読み込んだファイルを文字列ストリームに変換
      if (ss >> t_a >> t_val) {
        table_a.push_back(t_a);
        table_val.push_back(t_val);
      }
    }
    is_table_loaded = true;
  }

  // 仕様書4章 //

  /**
   * 仕様書式(12)の計算本体
   * 全反射を起こす時の最大のcosθの計算
   * @param phi は方位角[rad]
   * @param a はコア軸からの距離[mm]
   */
  double Calc_cosThetaMax(double phi, double a) {
    double sinphi = sin(phi);
    double sin2thetamax;    
    if (_n2 < 0) { // single cladding
      sin2thetamax = (1.0 - _n1*_n1/_n0/_n0)/(1 - a*a*sinphi*sinphi/_d1/_d1);
    } else {   // double cladding
      sin2thetamax = (1.0 - _n2*_n2/_n0/_n0)/(1 - a*a*sinphi*sinphi/_d1/_d1);
    }
    if (sin2thetamax > 1.0) return 0.0;
    double costhetamax = sqrt(1.0 - sin2thetamax);
    return costhetamax;
  }

  /*!
   * 仕様書式(11)の積分本体
   * 任意のaに対するTrapping Efficiency P(a)の計算。
   */
  double Calc_Pa(double a) {
    gsl_function F;
    SetA(a);
    F.function = f_integral_solidangle;
    F.params = this;
    double result, err;
    gsl_integration_workspace *local_w = gsl_integration_workspace_alloc(1000);
    gsl_integration_qag(&F, 0, 2.0*M_PI, 0, 1e-5, 1000, GSL_INTEG_GAUSS31, local_w, &result, &err);
    gsl_integration_workspace_free(local_w);
    _result = 0.5 - result/4/M_PI;
    _err = err/4/M_PI;  
    return 0.5 - result/4/M_PI;  
  }

  /**
   * 仕様書式(13)の積分本体
   * Trapping Efficiency P(a)をaで平均化する計算
   */
  double Calc_P_average_over_a() {
    gsl_function F;
    F.function = f_integral_average_over_a;
    F.params = this;
    double result, err;
    gsl_integration_workspace *local_w = gsl_integration_workspace_alloc(1000);
    gsl_integration_qag(&F, 0, _d1, 0, 1e-5, 1000, GSL_INTEG_GAUSS41, local_w, &result, &err);
    gsl_integration_workspace_free(local_w);
    _P = 2.0*result/_d1/_d1;
    _Perr = 2.0*err/_d1/_d1;
    return 2.0*result/_d1/_d1;
  }

  // 仕様書5章 //

  /*!
   * 仕様書式(19)の積分本体
   * 任意のzでのTrapping Effisciency P(a,z)のφ積分の項。μ積分の項はf_integral_phi_attで行われている。
   * @param z は検出側とは逆のファイバー端からの距離[mm]
   */
  double Calc_Paz(double a, double z) {
    gsl_function F;
    SetA(a);
    SetZ(z);  
    F.function = f_integral_phi_att;
    F.params = this;
    double result, err;
    gsl_integration_workspace *local_w = gsl_integration_workspace_alloc(1000);
    gsl_integration_qag(&F, 0, 2.0*M_PI, 0, 1e-3, 1000, GSL_INTEG_GAUSS31, local_w, &result, &err);
    gsl_integration_workspace_free(local_w);
    _result = result/4/M_PI;
    _err = err/4/M_PI;  
    return result/4/M_PI;  
  }

  /*!
   * 仕様書式(20)の積分本体
   * ファイバー軸方向に一様に光子が入射した時のTrapping Efficiencyのz平均
   */
  double Calc_Pa_average_over_z(double a) {
    SetA(a);
    gsl_function F;
    F.function = f_integral_average_over_z;
    F.params = this;
    double result, err;
    gsl_integration_workspace *local_w = gsl_integration_workspace_alloc(1000);
    gsl_integration_qag(&F, 0, _L, 0, 1e-3, 1000, GSL_INTEG_GAUSS31, local_w, &result, &err);
    gsl_integration_workspace_free(local_w);
    _result = result/_L;
    _err = err/_L;
    return result/_L;
  }

  /**
   * 仕様書式(21)の積分本体
   * ファイバー軸方向に一様に光子が入射した時のTrapping Efficiencyのz平均にabsorptionの効果を入れたもの。
   */
  double Calc_P_average_over_za() {
    gsl_function F;
    F.function = f_integral_average_over_za;
    F.params = this;
    double result, err;
    gsl_integration_workspace *local_w = gsl_integration_workspace_alloc(1000);
    gsl_integration_qag(&F, 0, _d1, 0, 1e-3, 1000, GSL_INTEG_GAUSS31, local_w, &result, &err);
    gsl_integration_workspace_free(local_w);
    _Patt = 2*result/_d1/_d1;
    _Patterr = 2*err/_d1/_d1;
    return 2*result/_d1/_d1;
  }

  // 仕様書6章1節 //

  /**
   * 仕様書式(22)の計算本体
   * dP/dθ = 1/2π sinθ ∮_Φ(θ)dΦの計算
   */
  double dPdTheta(double a, double theta) { 
    gsl_function F;
    F.function = check_refrection;
    F.params = this;
    SetA(a);
    SetTheta(theta);
    double result, err;
    gsl_integration_workspace *local_w = gsl_integration_workspace_alloc(1000);    
    gsl_integration_qag(&F, 0, 2*M_PI, 0, 1e-6, 1000, GSL_INTEG_GAUSS41, local_w, &result, &err);
    gsl_integration_workspace_free(local_w);
    return result/2/M_PI;
  }

  /**
   * 仕様書式(23)の計算本体
   * dP/dθをファイバー軸からの距離aで平均化したもの
   */
  double dPdTheta(double theta) {
    gsl_function F;
    F.function = f_integral_theta_dist_a;
    F.params = this;
    SetTheta(theta);
    double result, err;
    gsl_integration_workspace *local_w = gsl_integration_workspace_alloc(5000);
    gsl_integration_qag(&F, 0, _d1, 0, 1e-3, 5000, GSL_INTEG_GAUSS41, local_w, &result, &err);
    gsl_integration_workspace_free(local_w); 
    double c = cos(theta);
    double s = sin(theta);
    double att = 1.0 - exp(-_L/c/_Latt);
    return 2.0*result/_d1/_d1*s*c*_Latt*att/_L;
  }

  /**
   * 仕様書には不記載
   * Trapping Efficiencyのファイバー軸からの距離a分布
   */
  double dPda(double a) {
    return 2.0*M_PI*Calc_a_dist_average_over_psi(a)*Calc_Pa_average_over_z(a);
  }

  // 仕様書6章2節 //

  /**
   * 現在使用されていません
   */
  double dPdl(double theta) {
    gsl_function F;
    F.function = f_integral_theta_dist_a;
    F.params = this;
    SetTheta(theta);
    double result, err;
    gsl_integration_workspace *local_w = gsl_integration_workspace_alloc(1000);
    gsl_integration_qag(&F, 0, _d1, 0, 1e-5, 1000, GSL_INTEG_GAUSS41, local_w, &result, &err);
    gsl_integration_workspace_free(local_w);
    double c = cos(theta);
    double s = sin(theta);
    double att = c*_Latt - exp(-_L/c/_Latt)*(c*_Latt + _L);
    return 2.0*result/_d1/_d1*s*_Latt*att/_L/c;
  }

  // 仕様書6章3節 //
  /**
   * 仕様書式(30)の被積分項
   * Trapping Effisciencyの時間分布dP/dt
   */
  double dPdt(double theta, double a, double z){
    double c = GetLightSpeed(); 
    double c_core = c/_n0;
    SetTheta(theta);
    SetA(a);
    SetZ(z);
    double costheta = cos(theta);
    double L = GetL();
    double l = L - z;
    double dPdtheta = dPdTheta(a, theta);
    double att = exp(-l/costheta/_Latt);
    return c_core*costheta*costheta/l*dPdtheta*att;
  }

  /**
   * 仕様書式(31)の計算本体
   * Trapping Effisciencyの時間分布dP/dtをθで平均化したもの
   */
  double dPdt_a(double z, double theta){
    gsl_function F;
    SetZ(z);
    SetTheta(theta);
    F.function = f_integral_a_dist;
    F.params = this;
    double result, err;
    double pts[2] = {0.0, _d1};
    size_t npts = 2;
    gsl_integration_workspace *local_w = gsl_integration_workspace_alloc(5000);
    int status = gsl_integration_qagp(&F, pts, npts, 0, 1e-3, 5000, local_w, &result, &err);
    gsl_integration_workspace_free(local_w);
    if (status != GSL_SUCCESS) {
        // エラーなら0を返すか、ログを出すなどの処理
        printf("Warning: Integration did not converge in dPdt_a(z=%f, theta=%f)\n", z, theta);
        return 0.0; 
    }
    return 2.0*result/_d1/_d1;
  }

  /**
   * 仕様書には不記載
   * Transit timeを計算する式
   */
  double Transit_time(double theta, double z){
    double c = GetLightSpeed(); 
    double c_core = c/_n0;
    double costheta = cos(theta);
    double L = GetL();
    double l = (L - z)/costheta;
    return l/c_core;
  }

  /**
   * Komae's (18)
   */
  double GetAprime(double a, double z, double theta, double phi) {
    SetZ(z);
    SetTheta(theta);
    SetPhi(phi);
    return f_aprime(a, this);
  }

  /**
   * Derivative of (18) with respect to a, partial a'/partial a
   */
  // 最後の結果が初期条件aによってどのような影響を受けいているかを微分によって評価
  double GetAprime_deriv(double a, double z, double theta, double phi) {
    SetZ(z);
    SetTheta(theta);
    SetPhi(phi);
    gsl_function F;
    F.function = f_aprime;
    F.params = this;
    double result, err;
    gsl_deriv_forward(&F, a, 0.001, &result, &err);
    return result;
  }

  // ファイバーのabsorptionの効果を計算する
  /* @param psi Azimuthal angle with respect to the cross-section of the fiber*/
  //aの初期位置を決定する分布関数
  double Get_a_initial_distribution(double a) {
    // まだ読み込んでいなければ読み込む (Lazy Loading)
    if (!is_table_loaded) {
      PrepareTable();
    }

    // 範囲外チェック
    if (table_a.empty()) return 0.0;
    if (a <= table_a.front()) return table_val.front();
    if (a >= table_a.back()) return 0.0; // 範囲外は0とする場合

    // --- 二分探索と線形補間 (高速) ---
    // a 以上の最初のイテレータを見つける
    auto it = lower_bound(table_a.begin(), table_a.end(), a);
    
    size_t idx = distance(table_a.begin(), it);
    
    double x1 = table_a[idx - 1];
    double x2 = table_a[idx];
    double y1 = table_val[idx - 1];
    double y2 = table_val[idx];

    // 線形補間: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
    return y1 + (a - x1) * (y2 - y1) / (x2 - x1);
  }

  //Escap angle Distribution
  double Escape_angle_distribution(double theta, double a){
    double n_ice = 1.309;
    double n =  n_ice / GetN0();
    double weight_a = Get_a_initial_distribution(a);
    double weight_jac = n * sqrt(1 - sin(theta)*sin(theta) / (n*n))/cos(theta);
    double weight = weight_a * weight_jac;
    return weight;
  };

  double dPdTheta_escape(double theta) { 
    gsl_function F;
    F.function = f_integral_escape_dist;
    F.params = this;
    SetTheta(theta);
    double result, err;
    double epsabs = 1e-10; // 絶対誤差の許容値
    double epsrel = 1e-5;  // 相対誤差の許容値
    gsl_integration_workspace *local_w = gsl_integration_workspace_alloc(5000);
    gsl_integration_qag(&F, 0, _d1, epsabs, epsrel, 5000, GSL_INTEG_GAUSS41, local_w, &result, &err);
    gsl_integration_workspace_free(local_w);
    double c = cos(theta);
    double s = sin(theta);
    double att = 1.0 - exp(-_L/c/_Latt);
    return 2.0*result/_d1/_d1*s*c*_Latt*att/_L;
  };
};

#endif
