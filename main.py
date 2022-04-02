import data
import model

if __name__ == "__main__":
    ds = data.load_data()

    #model.train_classifier(ds)

    model.train_explanator(ds)

    #model.evaluate_classifier(ds)
