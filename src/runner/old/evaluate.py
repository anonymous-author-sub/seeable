from hydra.utils import instantiate
from sklearn.metrics import roc_auc_score

from src.utils import Logger


class Evaluate:
    def __init__(self, cfg):
        self.cfg = cfg

        # load detection model
        print("loading baseline detection model...")
        self.model = instantiate(self.cfg.model)

    def run_model(self, dir, json_file):
        img_names, prediction = self.model.run(dir, json_file)
        assert isinstance(img_names, list) and isinstance(img_names, list)
        return img_names, prediction

    def compute_roc_auc(self, pred_real, pred_fake) -> float:
        labels = [0] * len(pred_real) + [1] * len(pred_fake)
        prediction = pred_real + pred_fake
        score = roc_auc_score(labels, prediction)
        return score

    def run(self):
        print("Detecting real images ...")
        img_names_real, pred_real = self.run_model(
            self.cfg.img_dir_real, self.cfg.json_file_real
        )

        print("Detecting fake images ...")
        preds_fake = []
        for test in self.cfg.test_db:
            dir = self.cfg.img_dir + test
            json_file = self.cfg.img_dir + test + "_meta.json"
            img_names, pred_fake = self.run_model(dir, json_file)

            score = self.compute_roc_auc(pred_real, pred_fake)
            print(f"Detection score for {test} is {score}")

            # print(score)
            self.log.write(f"{test}\t{score}\n")
            preds_fake.extend(pred_fake)

        score_all = self.compute_roc_auc(pred_real, pred_fake)
        print(f"AUC score for all is {score_all}")
