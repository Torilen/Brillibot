FROM ubuntu:18.04
RUN sudo apt-update -y
RUN sudo apt-get install git -y
RUN sudo apt-get install python-pip python-dev
RUN git clone https://github.com/facebookresearch/ParlAI.git
RUN sudo apt-get install curl -y
RUN cd ParlAI
RUN python setup.py develop
RUN cd ..
RUN pip install transformers==2.5.1
RUN pip install 'git+https://github.com/rsennrich/subword-nmt.git#egg=subword-nmt'
RUN pip install werkzeug==0.16.1
RUN pip install pyOpenSSL
RUN pip install BeautifulSoup4
COPY . /docker_app
WORKDIR /docker_app
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
EXPOSE 5000 
CMD ["python", "app/webapp.py", "-t", "blended_skill_task", "-mf", "zoo:blender/blender_90M/model"]
