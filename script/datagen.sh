docker run -v ./data/:/app/out/ -t belval/trdg:latest trdg \
--count 500000 \
--length 1 \
--format 64 \
--skew_angle 10 --random_skew \
--thread_count 8 \
--output_dir out/dataset/vi \
--dict /app/out/Viet148k.txt \
--font_dir /app/out/font \
--name_format 2

docker run -v ./data/:/app/out/ -t belval/trdg:latest trdg \
--count 500000 \
--length 1 \
--format 64 \
--skew_angle 10 --random_skew \
--thread_count 8 \
--output_dir out/dataset/en \
--language "en" \
--name_format 2

docker run -v ./data/:/app/out/ -t belval/trdg:latest trdg \
--count 100000 --random_sequences \
--length 1 \
--format 64 \
--skew_angle 10 --random_skew \
--thread_count 8 \
--output_dir out/dataset/random \
--language "en" \
--name_format 2 