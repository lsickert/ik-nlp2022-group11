from data_analysis import data
import model

if __name__ == "__main__":
    ds = data.load_data()

    #model.train_classifier(ds)

    #model.train_explanator(ds)

    #model.evaluate_classifier(ds)

    model.train_bart_classifier(ds)

    #model.predict_single("Children smiling and waving at camera</s>The kids are frowning")
