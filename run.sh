echo "16node"
cp examples/sync/16worker.conf examples/sync/job.conf
./bin/singa-run.sh -workspace=examples/sync/

exit
echo "1node"
cp examples/sync/1worker.conf examples/sync/job.conf
./bin/singa-run.sh -workspace=examples/sync/
sleep 1
echo "2node"
cp examples/sync/2worker.conf examples/sync/job.conf
./bin/singa-run.sh -workspace=examples/sync/
sleep 1
echo "4node"
cp examples/sync/4worker.conf examples/sync/job.conf
./bin/singa-run.sh -workspace=examples/sync/
sleep 1
echo "8node"
cp examples/sync/8worker.conf examples/sync/job.conf
./bin/singa-run.sh -workspace=examples/sync/
sleep 1
sleep 1
echo "32node"
cp examples/sync/32worker.conf examples/sync/job.conf
./bin/singa-run.sh -workspace=examples/sync/
