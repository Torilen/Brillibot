FROM python:3.7
RUN apt-get update -y && apt-get install git python-pip python-dev build-essential curl -y 
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN git clone https://github.com/facebookresearch/ParlAI.git
WORKDIR ParlAI
RUN python setup.py develop
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir tokenizers==0.10.1 && \
    pip install --no-cache-dir transformers && \
    pip install --no-cache-dir 'git+https://github.com/rsennrich/subword-nmt.git#egg=subword-nmt' && \
    pip install --no-cache-dir werkzeug==0.16.1 && \
    pip install --no-cache-dir pyOpenSSL && \
    pip install --no-cache-dir BeautifulSoup4 
RUN ls
COPY . ../app
WORKDIR ../
WORKDIR ./app
RUN ls
RUN pip install --no-cache-dir -r requirements.txt && wget -O ./app/lib/stanford-postagger-full-2018-10-16.zip  https://nlp.stanford.edu/software/stanford-postagger-full-2018-10-16.zip && unzip ./app/lib/stanford-postagger-full-2018-10-16.zip 
ENTRYPOINT ["python"]
EXPOSE 5000 
CMD ["app/webapp.py", "-t", "blended_skill_task", "-mf", "zoo:blender/blender_90M/model"]
