#!/bin/bash

EXYNOS_IP=192.168.0.11

# If authorized_key exists, do nothing
if [ -e "/home/$USER/.ssh/id_rsa.pub" ]
then
    echo '[INFO] authorized_keys already exists in the host'
else
    ssh-keygen -t rsa
fi

# Move ssh id_rsa to Exynos board
DOES_KEY_EXIST=`ssh -q root@$EXYNOS_IP [[ -f '/home/root/.ssh/authorized_keys' ]] && echo "True" || echo "False";`
if [[ $DOES_KEY_EXIST == "True" ]];
then
    exit
else
    cat ~/.ssh/id_rsa.pub > ~/.ssh/_authorized_keys
    ssh root@$EXYNOS_IP 'mkdir /home/root/.ssh'
    scp ~/.ssh/_authorized_keys root@$EXYNOS_IP:/home/root/.ssh/authorized_keys
    rm ~/.ssh/_authorized_keys
fi