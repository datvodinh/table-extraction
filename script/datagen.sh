docker run -v ./data/:/app/out/ -t belval/trdg:latest trdg \
--count 300000 \
--length 1 \
--format 32 \
--skew_angle 10 --random_skew \
--thread_count 16 \
--output_dir out/dataset/vi \
--dict /app/out/vi.txt \
--font_dir /app/out/font \
--name_format 2

docker run -v ./data/:/app/out/ -t belval/trdg:latest trdg \
--count 600000 \
--length 1 \
--format 32 \
--skew_angle 10 --random_skew \
--thread_count 16 \
--dict /app/out/en.txt \
--output_dir out/dataset/en \
--name_format 2

docker run -v ./data/:/app/out/ -t belval/trdg:latest trdg \
--count 100000 --random_sequences \
--length 1 \
--format 32 \
--skew_angle 10 --random_skew \
--thread_count 16 \
--output_dir out/dataset/random \
--language "en" \
--name_format 2 

docker run -v ./data/:/app/out/ -t belval/trdg:latest trdg \
--count 200000 \
--length 1 \
--format 32 \
--skew_angle 10 --random_skew \
--thread_count 16 \
--output_dir out/dataset/metadata \
--dict /app/out/metadata.txt \
--font_dir /app/out/font \
--name_format 2