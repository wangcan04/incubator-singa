/**************************************************************
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/

#include "singa/utils/confusion.h"

namespace singa{

ConfusionMatrix::ConfusionMatrix(int nclasses): nclasses_(nclasses){
  for (int i=0; i<nclasses_; i++){
    matrix_.push_back(vector<int>()); 
    for (int j=0; j<nclasses_; j++)
      matrix_[i].push_back(0); 
  }
  Reset();
}

void ConfusionMatrix::Reset(){
  for (int i=0; i<nclasses_; i++)
    for (int j=0; j<nclasses_; j++)
      matrix_[i][j] = 0; 
}

void ConfusionMatrix::Add(int target, int predict){
  matrix_[target][predict]++; 
}

double ConfusionMatrix::precision(){
  int count = matrix_[1][1]+matrix_[0][1];
  if (nclasses_==2 && count>0)
    return 1.0*matrix_[1][1]/(count);
  return 1; 
}
double ConfusionMatrix::recall(){
  int count = matrix_[1][1]+matrix_[1][0];
  if (nclasses_==2 && count > 0)
    return 1.0*matrix_[1][1]/(count);
  return 1; 
}

string ConfusionMatrix::ToString(){
  string output = "Confusion matrix: [true labels][predicted]";
  double row_valids = 0.0; 
  for (int i=0; i< nclasses_; i++){
    output+="\n";
    int total_row=0; 
    for (int j=0; j<nclasses_; j++){
      output = output + "\t" + std::to_string(matrix_[i][j]); 
      total_row+=matrix_[i][j];
    }
    row_valids += ((1.0)*matrix_[i][i]/total_row); 
  }
  output = output + "\n Average row accuracy = " + std::to_string(row_valids/nclasses_);
  if (nclasses_==2){
    double precision = 1.0*matrix_[1][1]/(matrix_[1][1]+matrix_[0][1]);
    double recall = 1.0*matrix_[1][1]/(matrix_[1][1]+matrix_[1][0]);
    double f1 = 2*(precision*recall)/(precision+recall);
    output = output +"\n Precision = " + std::to_string(precision);
    output = output +"\n Recall = " + std::to_string(recall);
    output = output +"\n F1 = " + std::to_string(f1);
  }
  return output; 
}
} // end namespace
