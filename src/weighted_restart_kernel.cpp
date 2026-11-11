// Streaming lambda/restart mixture. Each candidate has a shared numerical scale.
#include <vector>
#include <cmath>
#include <algorithm>
extern "C" int weighted_grid(int n, int m, int k, const double* y,
 const double* u, const double* lambda, const double* weights, double total,
 double alpha, const int* oracle_index, int ne, const int* eps_index,
 double* upper, unsigned char* covered, int* stopping) {
 std::vector<double> capital(m*k,0.0),scale(m,0.0);
 std::vector<unsigned char> rejected(m);
 double reserve=total;
 for(int e=0;e<ne;++e) stopping[e]=n+1;
 for(int t=0;t<n;++t){
  reserve-=weights[t];
  for(int j=0;j<m;++j){
   double* a=&capital[j*k];
   double add=weights[t]*std::exp(-scale[j]), sum=0, mx=0;
   for(int l=0;l<k;++l){
    a[l]=(a[l]+add)*(1+lambda[l]*(u[j]-y[t]));
    sum+=a[l]; mx=std::max(mx,a[l]);
   }
   if(!std::isfinite(sum)) return 1;
   double logvalue=sum>0 ? std::log(sum/k)+scale[j] : -INFINITY;
   rejected[j]=logvalue>std::log(1/alpha-reserve);
   if(mx>1e150){for(int l=0;l<k;++l) a[l]*=1e-100; scale[j]+=100*std::log(10.0);}
   else if(mx>0 && mx<1e-100 && scale[j]>0){for(int l=0;l<k;++l) a[l]*=1e100; scale[j]-=100*std::log(10.0);}
  }
  upper[t]=1;
  for(int j=0;j<m;++j) if(rejected[j]){upper[t]=u[j];break;}
  covered[t]=!rejected[oracle_index[t]];
  for(int e=0;e<ne;++e) if(stopping[e]==n+1 && rejected[eps_index[e]]) stopping[e]=t+1;
 }
 return 0;
}
