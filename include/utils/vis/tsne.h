/*
 *
 * Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */


#ifndef TSNE_H
#define TSNE_H


static inline float sign(float x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }


class TSNE
{
public:
    void run(int N, int D, float* X, float *Y){
    void run(float* X, int N, int D, float* Y, int no_dims, float perplexity, float theta);
    bool load_data(float** data, int* n, int* d, int* no_dims, float* theta, float* perplexity, int* rand_seed);
    void save_data(float* data, int* landmarks, float* costs, int n, int d);
    void symmetrizeMatrix(unsigned int** row_P, unsigned int** col_P, float** val_P, int N); // should be static!


private:
    void computeGradient(float* P, unsigned int* inp_row_P, unsigned int* inp_col_P, float* inp_val_P, float* Y, int N, int D, float* dC, float theta);
    void computeExactGradient(float* P, float* Y, int N, int D, float* dC);
    float evaluateError(float* P, float* Y, int N, int D);
    float evaluateError(unsigned int* row_P, unsigned int* col_P, float* val_P, float* Y, int N, int D, float theta);
    void zeroMean(float* X, int N, int D);
    void computeGaussianPerplexity(float* X, int N, int D, float* P, float perplexity);
    void computeGaussianPerplexity(float* X, int N, int D, unsigned int** _row_P, unsigned int** _col_P, float** _val_P, float perplexity, int K);
    void computeSquaredEuclideanDistance(float* X, int N, int D, float* DD);
    float randn();
};

#endif

