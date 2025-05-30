FROM containers.intersystems.com/intersystems/irishealth-community:2025.1

#USER irisowner

ENV PYTHON_PATH=/usr/irissys/bin/
ENV LD_LIBRARY_PATH=${ISC_PACKAGE_INSTALLDIR}/bin:${LD_LIBRARY_PATH}
ENV PATH="/home/irisowner/.local/bin:/usr/irissys/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/home/irisowner/bin"

RUN find . -name '.DS_Store' -type f -delete

WORKDIR /irisdev
COPY --chown=irisowner:irisowner --chmod=700 init /irisdev/init
COPY --chown=irisowner:irisowner --chmod=700 src /irisdev/src

WORKDIR /irisdev/init

#RUN pip config set global.break-system-packages true
#RUN pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN iris start IRIS \
    && iris session IRIS < iris.script \
    && iris stop IRIS quietly
