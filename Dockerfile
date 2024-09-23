FROM binarii/faster_liveportrait:v3 

COPY . .

RUN <<EOF
rm -rf dist
EOF

CMD ["/bin/bash", "-l", "entrypoint.sh"]

