FROM nvidia/cuda:10.2-base
CMD nvidia-smi
RUN apt-get update -y && apt-get install python3.7 git python3-pip python3.7-dev build-essential curl default-jre default-jdk python3-setuptools -y
RUN python3.7 -m pip install --upgrade pip
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN git clone https://github.com/facebookresearch/ParlAI.git
RUN python3.7 -m pip install googletrans==3.1.0a0
WORKDIR ParlAI
RUN git reset --hard 0107e74d83b662e347890808dab02ef658d9e254 && python3.7 setup.py develop
RUN echo $PYTHONPATH
RUN python3.7 -m pip install --no-cache-dir -r requirements.txt
RUN python3.7 -m pip install --no-cache-dir --upgrade pip && \
    python3.7 -m pip install --no-cache-dir tokenizers==0.10.1 && \
    python3.7 -m pip install --no-cache-dir tensorflow==2.4.1 && \
    python3.7 -m pip install --no-cache-dir Flask && \
    python3.7 -m pip install --no-cache-dir flask-restx && \
    python3.7 -m pip install --no-cache-dir flask-cors && \
    python3.7 -m pip install --no-cache-dir transformers && \
    python3.7 -m pip install --no-cache-dir pandas==1.1.5 && \
    python3.7 -m pip install --no-cache-dir nltk==3.2.5 && \
    python3.7 -m pip install --no-cache-dir keras==2.4.3 && \
    python3.7 -m pip install --no-cache-dir torch==1.5.0 && \
    python3.7 -m pip install --no-cache-dir 'git+https://github.com/rsennrich/subword-nmt.git#egg=subword-nmt' && \
    python3.7 -m pip install --no-cache-dir werkzeug==2.0.1 && \
    python3.7 -m pip install --no-cache-dir pyOpenSSL && \
    python3.7 -m pip install --no-cache-dir BeautifulSoup4 && \
    python3.7 -m pip install --no-cache-dir joblib && \
    python3.7 -m pip install --no-cache-dir scikit-learn && \
    python3.7 -m pip uninstall pyyaml -y && \
    python3.7 -m pip install --no-cache-dir google-cloud && \
    python3.7 -m pip install --no-cache-dir google-cloud-translate && \
    python3.7 -m pip install --no-cache-dir scipy==1.4.1 && \
    python3.7 -m pip install --no-cache-dir hdbscan
RUN ls
COPY . ../app
WORKDIR ../
WORKDIR ./app
RUN ls
ENTRYPOINT ["python3.7"]
EXPOSE 5000
CMD ["app/webapp.py", "-t", "blended_skill_talk", "-mf", "zoo:blender/blender_90M/model", "--fp16", "false"]
