default: target svmlight svmlight_lsi300

clean:
	rm -f data/text/target-train.csv data/bag-of-words/*.svmlight data/bag-of-words/*.svmlight.index

target:
	paste -d ',' data/text/text-files-train.dat data/text/target-train.dat > data/text/target-train.csv

svmlight:
	python converters/text2svmlight.py data/text/text-files-train.dat data/text/text-files-test.dat -o data/bag-of-words/train_tfidf.svmlight data/bag-of-words/test_tfidf.svmlight -l data/text/target-train.csv --filter_threshold 2 -v

svmlight_lsi300:
	python converters/text2svmlight.py data/text/text-files-train.dat data/text/text-files-test.dat -o data/bag-of-words/train_tfidf_lsi300.svmlight data/bag-of-words/test_tfidf_lsi300.svmlight -l data/text/target-train.csv --filter_threshold 2 --lsi_topic 300 -v
