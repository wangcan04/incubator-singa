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



#ifndef SINGA_UTILS_CONFUSION_
#define SINGA_UTILS_CONFUSION_

#include <vector>
#include <string>

using std::string;
using std::vector; 
namespace singa{

/**
 * Confusion matrix provides more details about the model's predictions. 
 * It is essentially a two dimensional array [NCLASSES][NCLASSES] where
 * the first dimension is the actual label (target), and the second the
 * predicted label. 
 */

class ConfusionMatrix{
  public: 
    ConfusionMatrix(int nclasses);

    // Update the matrix with a target and a predicted label
    void Add(int target, int predict); 

    // Clear out the content
    void Reset(); 

    // Print out the content
    string ToString();  

  private:
    int nclasses_; // number of classes
    vector<vector<int>> matrix_; 
};

} // end namespace

#endif
