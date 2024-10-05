#!/bin/bash

PORT=22015
MACHINE=paffenroth-23.dyn.wpi.edu

ssh -i david_key -p ${PORT} -o StrictHostKeyChecking=no student-admin@${MACHINE}
