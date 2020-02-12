// this code should work for multi dimensional data

functions {

  real logmultiNormKro(matrix G, matrix S,int n,int m, matrix Sigma){
    matrix[m,n] tG;
    matrix[n,n] H;
    matrix[m,m] Gt0;
    matrix[m,m] Gt1;
    tG=G';
    H=diag_matrix(rep_vector(1.0,n))-1.0/n*rep_matrix(1.0,n,n);
    Gt0=tG*H*tG';
    Gt1=tG*tG';
    return (-0.5*log_determinant(Sigma + n*S)-0.5*(n-1)*log_determinant(Sigma) -0.5*trace(n*(Gt0 + square(n) *S)\(Gt1 + square(n)*S)));
    }
}
data {
  int<lower=0> n; //number of data points
  int<lower=0> m; //number of inducing points
  int<lower=1> d; //number of dimensions
  matrix[n,d] x;//first data set
  matrix[n,d] y;// second data set
  matrix[m,d] z; // inducing points
}

transformed data {
  matrix[n,m] xz_dist2;
  matrix[n,m] yz_dist2;
  matrix[n,m] prod_xz_yz_dist;
  matrix[m,m] z_dist2;


  for (i in 1:n){
    for (j in 1:m){
        xz_dist2[i, j] = squared_distance(x[i],z[j]);
        yz_dist2[i, j] = squared_distance(y[i],z[j]);
    }
  }

  for (i in 1:m){
    for(j in 1:m){
        z_dist2[i, j] = squared_distance(z[i], z[j]);
    }
  }
}

parameters {
  real<lower=0> theta;
}

transformed parameters {
  matrix[m,m] R;
  matrix[n,m] Kxz;
  matrix[n,m] Kyz;
  matrix[n,m] G; // difference of the features
  matrix[m,m] Sigma;
  real volJ;
  real ml;

  matrix[m, d] Jx;
  matrix[m, d] Jy;
  matrix[m, 2*d] J;
  vector[n] logdetJTJ;
  vector[2*d] sJ;


  R = exp(- z_dist2/(4*theta^2)) + diag_matrix(rep_vector(10^(-8), m));
  Kxz = exp(- xz_dist2/(2*theta^2));
  Kyz = exp(- yz_dist2/(2*theta^2));
  G = Kxz-Kyz;


  // for Jacobian
  for (ii in 1:n) {
    for (ll in 1:m) {
        Jx[ll] = (-1./theta^2) * Kxz[ii, ll] * (x[ii] - z[ll]);
        Jy[ll] = (1./theta^2) * Kyz[ii, ll] * (y[ii] - z[ll]);
    }
    J = append_col(Jx, Jy);
    sJ = singular_values(J);
    logdetJTJ[ii] = sum(log(square(sJ)));
  }

  volJ = 0.5 * sum(logdetJTJ);



  // empirical covariance
  Sigma = 1.0/n*G'*(diag_matrix(rep_vector(1.0,n))-1.0/n*rep_matrix(1.0,n,n))*G+diag_matrix(rep_vector(0.1,m));

  ml=logmultiNormKro(G, R, n, m, Sigma);
  //print("theta=", theta, "-- volJ=", volJ, "--ml=",ml)


}


model {
  target += volJ;
  target += ml;

  theta ~ gamma(2,2);
}






