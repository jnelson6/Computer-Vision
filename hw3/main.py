
import histclassifier


if __name__ == '__main__':
    bins_set = [4, 8, 16, 32]
    for bins in bins_set:
        imglist = histclassifier.scan_images(['ImClass/', ])
        df_img = histclassifier.compute_multihist(imglist, bins, True)
        for k in [1, 3]:
            actual, predicted, acc = histclassifier.classify(df_img, bins, k)
            print("\n\n==============================")
            print("Classification using {} bins:".format(bins))
            print("------------------------------")
            actual = actual.reset_index()
            for index, row in actual.iterrows():
                # print(index, row)
                print("Test image {} of class <{}> has been assigned to class <{}>".format(
                    int(row['id']), row['label'], predicted[index]
                ))
            print("~~~~~~~~~~~~~~~~~~~~~")
            print("Test Accuracy(bins={}, k={}): {:.2f}%".format(
                bins, k, 100 * acc))
