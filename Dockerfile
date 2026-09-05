FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python3 -c "import duckdb; con = duckdb.connect(); con.execute('INSTALL httpfs;')" # Run tu dau de khoi bi loi mat song

COPY . .

CMD ["bash"]