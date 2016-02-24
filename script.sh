#!/bin/bash

./bin/singa-run.sh -conf examples/anh/job.conf >results/out_oversampled_b32_1 2>&1
sleep 5
./bin/singa-run.sh -conf examples/anh/job.conf >results/out_oversampled_b32_2 2>&1
sleep 5
./bin/singa-run.sh -conf examples/anh/job.conf >results/out_oversampled_b32_3 2>&1
sleep 5

./bin/singa-run.sh -conf examples/anh/job_noskip.conf >results/out_oversampled_b32_noskip_1 2>&1
sleep 5
./bin/singa-run.sh -conf examples/anh/job_noskip.conf >results/out_oversampled_b32_noskip_2 2>&1
sleep 5
./bin/singa-run.sh -conf examples/anh/job_noskip.conf >results/out_oversampled_b32_noskip_3 2>&1
sleep 5

./bin/singa-run.sh -conf examples/anh/job_b64.conf >results/out_oversampled_b64_1 2>&1
sleep 5
./bin/singa-run.sh -conf examples/anh/job_b64.conf >results/out_oversampled_b64_2 2>&1
sleep 5
./bin/singa-run.sh -conf examples/anh/job_b64.conf >results/out_oversampled_b64_3 2>&1
sleep 5

./bin/singa-run.sh -conf examples/anh/job_b64_noskip.conf >results/out_oversampled_b64_noskip_1 2>&1
sleep 5
./bin/singa-run.sh -conf examples/anh/job_b64_noskip.conf >results/out_oversampled_b64_noskip_2 2>&1
sleep 5
./bin/singa-run.sh -conf examples/anh/job_b64_noskip.conf >results/out_oversampled_b64_noskip_3 2>&1
sleep 5

./bin/singa-run.sh -conf examples/anh/job_b96.conf >results/out_oversampled_b96_1 2>&1
sleep 5
./bin/singa-run.sh -conf examples/anh/job_b96.conf >results/out_oversampled_b96_2 2>&1
sleep 5
./bin/singa-run.sh -conf examples/anh/job_b96.conf >results/out_oversampled_b96_3 2>&1
sleep 5

