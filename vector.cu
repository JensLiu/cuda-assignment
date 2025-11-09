void InitV(int N, double *v) {
   int i;
   //srand(17);
   for (i=0; i<N; i++) 
     v[i] = (double) rand() / (double) RAND_MAX;
     //v[i] = (double) rand();
     //v[i] = 2.5;
   
}

void InitV_float(int N, float *v) {
   int i;
   //srand(17);
   for (i=0; i<N; i++) 
     v[i] = (float) rand() / (float) RAND_MAX;
     //v[i] = (double) rand();
     //v[i] = 2.5;
   
}

int error(double a, double b) {
  double tmp;

  tmp = abs(a-b) / abs(a);

  if (tmp > 0.0001) return 1;
  else  return 0;

}

int Test(int N, double *v, double sum, double *res) {
   int i;
   double tmp;

   tmp = 0.0;
   for (i=0; i<N; i++) 
     tmp = tmp + v[i];
   *res = tmp;
   if (error(tmp, sum)) return 0;
   return 1;
}


int Test_float(int N, float *v, float sum, float *res) {
   int i;
   float tmp;

   tmp = 0.0;
   for (i=0; i<N; i++) 
     tmp = tmp + v[i];
   *res = tmp;
   if (error(tmp, sum)) return 0;
   return 1;
}
