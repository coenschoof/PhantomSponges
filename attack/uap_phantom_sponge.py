
import json #import pickle
import random
from pathlib import Path
import torch
import numpy as np
import torchvision
from torchvision import transforms
import torch.nn as nn

from local_yolos.yolov5.utils.general import non_max_suppression, xyxy2xywh
from attacks_tools.early_stopping_patch import EarlyStopping

transt = transforms.ToTensor()
transp = transforms.ToPILImage()

def get_model(name):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if name == 'yolov5':
        # taken from https://github.com/ultralytics/yolov5
        from local_yolos.yolov5.models.experimental import attempt_load
        model = attempt_load('local_yolos/yolov5/weights/yolov5s.pt', device).eval()
    elif name == 'yolov4':
        # taken from https://github.com/WongKinYiu/PyTorch_YOLOv4
        from local_yolos.yolov4.models.models import Darknet, load_darknet_weights
        model = Darknet('local_yolos/yolov4/cfg/yolov4.cfg', img_size=640).to(device).eval()
        load_darknet_weights(model, 'local_yolos/yolov4/weights/yolov4.weights')
    elif name == 'yolov3':
        # taken from https://github.com/ultralytics/yolov3
        from local_yolos.yolov3 import hubconf
        model = hubconf.yolov3(pretrained=True, autoshape=False, device=device)
    return model

#vergelijkbaar met een neural network, heeft ook een forward functie
class IoU(nn.Module):
    def __init__(self, conf_threshold, iou_threshold, img_size, device) -> None:
        #callt de constructor van de superclass, in dit geval nn.Module
        super(IoU, self).__init__()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.device = device

    def forward(self, output_clean, output_patch):
        batch_loss = []



        #tensor([640, 640, 640, 640])
        gn = torch.tensor(self.img_size)[[1, 0, 1, 0]]
        gn = gn.to(self.device)
        #returnt: (each list idx (tensor) is an image, each row is a detection)
        # pred_clean_bboxes = [
        #     torch.tensor([
        #         [10, 20, 50, 70, 0.9, 2],  # [x1, y1, x2, y2, object_confidence, class_label]
        #         [30, 40, 80, 100, 0.8, 1],
        #         [60, 70, 120, 160, 0.95, 3],
        #     ]),
        #     torch.tensor([
        #         [20, 30, 70, 90, 0.88, 1],
        #         [40, 50, 90, 110, 0.75, 2],
        #     ])
        # ]
        pred_clean_bboxes = non_max_suppression(output_clean, self.conf_threshold, self.iou_threshold, classes=None,
                                                max_det=1000)
        #WAAROM NEEM JE INEENS EEN CONF VAN 0.001
        patch_conf = 0.001
        #print(   torch.sum(output_patch[:,: , 4:5].view(-1) > 0.25)   )
        #WAAROM OPTIMALISEREN WE DE DIRTY OUTPUT OM EEN CONFWAARDE HOGER DAN 0.25 TE HEBBEN, 
        # TERWIJL WE HIER SLECHTS EEN CONFWAARDE AANNEMEN VAN 0.001?
        pred_patch_bboxes = non_max_suppression(output_patch, patch_conf, self.iou_threshold, classes=None,
                                                max_det=30000)

        # print final amount of predictions
        #onderstaande code wordt niks mee gedaan?
        final_preds_batch = 0
        for img_preds in non_max_suppression(output_patch, self.conf_threshold, self.iou_threshold, classes=None,
                                             max_det=30000):
            final_preds_batch += len(img_preds)

        #loop door alle tensors (dus ieder plaatje)
        for (img_clean_preds, img_patch_preds) in zip(pred_clean_bboxes, pred_patch_bboxes):  # per image
            #loop door alle detections, de cleane
            for clean_det in img_clean_preds:
                #neem de class van de huidige clean detection (bijv. car)
                clean_clss = clean_det[5]
                #
                clean_xyxy = torch.stack([clean_det])  # .clone()
                #neemt alleen de xyxy values en deelt ieder door 640
                #dus iets als torch.tensor([[10/640, 20/640, 50/640, 70/640]])
                clean_xyxy_out = (clean_xyxy[..., :4] / gn).to(
                    self.device)
                
                #neem alle detections van het huidige patch plaatje waarbij de clean class en de adv class met elkaar overeenkomen
                #dit resulteert dus in een set patch_bboxes detections waarvan de class car is
                img_patch_preds_out = img_patch_preds[img_patch_preds[:, 5].view(-1) == clean_clss]
                #neem de xyxy waarden van bovenstaande, en deel het door 640
                #     torch.tensor([
                #         [10/640, 20/640, 50/640, 70/640],  # [x1, y1, x2, y2]
                #         [30/640, 40/640, 80/640, 100/640],
                #         [60/640, 70/640, 120/640, 160/640],
                #     ]),
                patch_xyxy_out = (img_patch_preds_out[..., :4] / gn).to(self.device)

                if len(clean_xyxy_out) != 0:
                    #returnt een lijst van bijv 3x1 wanneer er 3 adversarial bbs zijn, en 1 clean bb.
                    #Elke index in deze tensor is een IoU score
                    target = self.get_iou(patch_xyxy_out, clean_xyxy_out)
                    if len(target) != 0:
                        #Pak de maximum
                        target_m, _ = target.max(dim=0)
                    else:
                        #return 0
                        target_m = torch.zeros(1).to(self.device)
                    #voeg de max aan een lijst toe
                    batch_loss.append(target_m)

        one = torch.tensor(1.0).to(self.device)
        if len(batch_loss) == 0:
            return one
        #Zie eq. 9 in paper
        return (one - torch.stack(batch_loss).mean())

    def get_iou(self, bbox1, bbox2):
        """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
        is simply the intersection over union of two boxes.  Here we operate on
        ground truth boxes and default boxes.
        E.g.:
            A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        Args:
            bbox1: (tensor) Ground truth bounding boxes, Shape: [num_objects, 4]
            bbox2: (tensor) Prior boxes from priorbox layers, Shape: [num_priors, 4]
        Return:
            jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
        """

        inter = self.intersect(bbox1, bbox2)
        area_a = ((bbox1[:, 2] - bbox1[:, 0]) *
                  (bbox1[:, 3] - bbox1[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((bbox2[:, 2] - bbox2[:, 0]) *
                  (bbox2[:, 3] - bbox2[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        union = area_a + area_b - inter
        return inter / union

    def intersect(self, box_a, box_b):
        """ We resize both tensors to [A,B,2] without new malloc:
        [A,2] -> [A,1,2] -> [A,B,2]
        [B,2] -> [1,B,2] -> [A,B,2]
        Then we compute the area of intersect between box_a and box_b.
        Args:
          box_a: (tensor) bounding boxes, Shape: [A,4].
          box_b: (tensor) bounding boxes, Shape: [B,4].
        Return:
          (tensor) intersection area, Shape: [A,B].
        """
        A = box_a.size(0) #always 1 since we loop over each clean_det
        B = box_b.size(0) #variable length since we compare the single clean_det against multiple adv detections
        #eerste argument in onderstaande regel
        #box_a[:, 2:] = neem x2 en y2
        #.unsqueeze(1) = voeg extra brackets toe zodat tensor([[[0.0781, 0.1094]]])
        #.expand(A, B, 2) = expand zodat tensor([[[0.0781, 0.1094],
                                                #[0.0781, 0.1094],
                                                #[0.0781, 0.1094]]])
        #hetzelfde voor de tweede argument tensor([[[0.0781, 0.1094],
                                                    #[0.1250, 0.1562],
                                                    #[0.1875, 0.2500]]])
        #neem uiteindelijk de minimum tussen deze 2 tensors, dus de [[0.0781, 0.1094.... 
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), 
                           box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                           box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]

class UAPPhantomSponge:
    def __init__(self, patch_folder, train_loader, val_loader, epsilon=0.1, iter_eps=0.05, penalty_regularizer=0,
                 lambda_1=0.75, lambda_2=0, use_cuda=True, epochs=70, patch_size=[640, 640], models_vers=[5]):

        self.use_cuda = use_cuda and torch.cuda.is_available()
        print("CUDA Available: ", self.use_cuda)
        self.device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

        self.train_loader = train_loader
        self.val_loader = val_loader

        # load wanted models
        self.models = []
        if 3 in models_vers:
          self.models.append(get_model('yolov3'))
        if 4 in models_vers:
          self.models.append(get_model('yolov4'))
        if 5 in models_vers:
          self.models.append(get_model('yolov5'))

        self.iter_eps = iter_eps
        self.penalty_regularizer = penalty_regularizer

        self.epsilon = epsilon
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

        self.epochs = epochs
        self.patch_size = patch_size
        #Deze functie handelt zowel de NMS als de IoU af
        self.iou = IoU(conf_threshold=0.25, iou_threshold=0.45, img_size=patch_size, device=self.device)

        self.full_patch_folder = "uap_train/" + patch_folder + "/"
        Path(self.full_patch_folder).mkdir(parents=True, exist_ok=False)

        self.current_dir = "experiments/" + patch_folder
        self.create_folders()

        self.current_train_loss = 0.0
        self.current_max_objects_loss = 0.0
        self.current_min_bboxes_added_preds_loss = 0.0
        self.current_orig_classification_loss = 0.0

        self.train_losses = []
        self.max_objects_loss = []
        self.min_bboxes_added_preds_loss = []
        self.orig_classification_loss = []

        self.val_losses = []
        self.val_max_objects_loss = []
        self.val_min_bboxes_added_preds_loss = []
        self.val_orig_classification_loss = []

        self.writer = None

    def create_folders(self):
        Path('/'.join(self.current_dir.split('/')[:2])).mkdir(parents=True, exist_ok=True)
        Path(self.current_dir).mkdir(parents=True, exist_ok=True)
        Path(self.current_dir + '/final_results').mkdir(parents=True, exist_ok=True)
        Path(self.current_dir + '/saved_patches').mkdir(parents=True, exist_ok=True)
        Path(self.current_dir + '/losses').mkdir(parents=True, exist_ok=True)
        Path(self.current_dir + '/testing').mkdir(parents=True, exist_ok=True)

    #wordt gecalled zodra alle batches zijn behandeld van 1 epoch (aan het einde van elke epoch dus)
    #Legt het volgende vast (wat ook daadwerkelijk elders gebruikt wordt):
        #train_losses = alle losses (obv training set) bij elkaar, gemiddelde over de epoch
        #max_objects_loss = de gemiddeld obj_loss over de epoch
        #orig_classification_loss = de gemiddelde IoU loss over de epoch
        #val_losses = alle losses bij elkaar obv de validation set
    #Doet dus in principe de eindresultaten van een epoch vastleggen, oa de validation loss
    def last_batch_calc(self, adv_patch, epoch_length,
                        epoch, i_batch):
        # calculate epoch losses

        #gemiddelde loss van obj_loss + bb_loss + IoU_loss over alle batches
        self.current_train_loss /= epoch_length
        #gemiddelde loss van obj_loss over alle batches
        self.current_max_objects_loss /= epoch_length
        #gemiddlede loss van de bb_loss over alle batches
        self.current_min_bboxes_added_preds_loss /= epoch_length
        #gemiddelde loss van IoU_loss over alle batches
        self.current_orig_classification_loss /= epoch_length

        #aan het eind van elke epoch losses vastleggen in een lijst
        self.train_losses.append(self.current_train_loss)
        self.max_objects_loss.append(self.current_max_objects_loss)
        self.min_bboxes_added_preds_loss.append(self.current_min_bboxes_added_preds_loss)
        self.orig_classification_loss.append(self.current_orig_classification_loss)

        self.train_results_dict = {}
        self.train_results_dict["train_losses"] = self.train_losses
        self.train_results_dict["train_max_objects_losses"] = self.max_objects_loss
        self.train_results_dict["train_min_bboxes_added_preds_losses"] = self.min_bboxes_added_preds_loss
        self.train_results_dict["train_orig_classification_losses"] = self.orig_classification_loss

        # check on validation

        #val_loss is de loss van de 3 termen bij elkaar
        #sep_val_loss is [max_objects_loss, min_bboxes_added_preds_loss, orig_classification_loss]
        val_loss, sep_val_loss = self.evaluate_loss(self.val_loader, adv_patch)
        #deze is nodig voor de early stopping
        self.val_losses.append(val_loss)

        self.val_results_dict = {}
        self.val_results_dict["val_losses"] = sep_val_loss[0]
        self.val_results_dict["val_max_objects_losses"] = sep_val_loss[0]
        self.val_results_dict["val_min_bboxes_added_preds_losses"] = sep_val_loss[1]
        self.val_results_dict["val_orig_classification_losses"] = sep_val_loss[2]

    def save_final_objects(self, adv_patch):
        # save patch
        transforms.ToPILImage()(adv_patch).save(
            self.current_dir + '/final_results/final_patch.png', 'PNG')

        # save losses
        with open(self.current_dir + '/losses/train_losses', 'w') as fp:
            json.dump(self.train_results_dict, fp)
        with open(self.current_dir + '/losses/val_losses', 'w') as fp:
            json.dump(self.val_results_dict, fp)


    def evaluate_loss(self, loader, adv_patch):
        print("Now determining the loss on the validation set")
        val_loss = []
        max_objects_loss = [] 
        orig_classification_loss = [] #max IoU
        min_bboxes_added_preds_loss = []
        #640x640
        adv_patch = adv_patch.to(self.device)

        for i, (img_batch, lab_batch, _) in enumerate(loader):
            #indien een ensemble van modellen is meegegeven, pak dan een random model
            r = random.randint(0, len(self.models) - 1)
            #geen backpropagation wordt gedaan nu, dus de gradient is niet nodig
            #loss.backward() rekent per parameter uit wat alle gradients zijn, 
            #maar we zijn toch alleen maar geinteresseerd in de gradients tov de input, niet al die w's
            with torch.no_grad():
                #Maakt van een tuple van 8 images een torch van 8x3x640x640
                img_batch = torch.stack(img_batch)
                img_batch = img_batch.to(self.device)
                #clampen tussen 0 en 1 want de clean images zijn ook tussen 0 en 1
                applied_batch = torch.clamp(img_batch[:] + adv_patch, 0, 1)
                with torch.no_grad():
                    output_clean = self.models[r](img_batch)[0]
                    output_patch = self.models[r](applied_batch)[0]
                
                
                max_objects = self.max_objects(output_patch)
                bboxes_area = self.bboxes_area(output_clean, output_patch)
                #print("evaluate_loss ",output_patch[..., 4][0])
                iou = self.iou(output_clean, output_patch)

                #eerste term uit de loss functie: max objects
                batch_loss = max_objects.item() * self.lambda_1
                max_objects_loss.append(max_objects.item() * self.lambda_1)

                #tweede term uit de loss functie: bbox area
                if not torch.isnan(bboxes_area):
                    batch_loss += (bboxes_area * self.lambda_2)
                    min_bboxes_added_preds_loss.append(bboxes_area.item() * self.lambda_2)

                #derde term uit de loss functie: max IoU
                if not torch.isnan(iou):
                    batch_loss += (iou.item() * (1 - self.lambda_1))
                    #1 - lambda_1 = lambda_3
                    orig_classification_loss.append(iou.item() * (1 - self.lambda_1)) 

                val_loss.append(batch_loss)

                print(f"combined_loss, max_obj, bb_area, iou_loss for batch {i+1}/{len(loader)} = \
                      {round(batch_loss.item(),3)}|{round(max_objects.item(),3)}|{round(bboxes_area.item(),3)}|{round(iou.item(),3)}")

                del img_batch, lab_batch, applied_batch, output_patch, batch_loss
                torch.cuda.empty_cache()

        loss = sum(val_loss) / len(val_loss)
        max_objects_loss = sum(max_objects_loss) / len(max_objects_loss)
        min_bboxes_added_preds_loss = sum(min_bboxes_added_preds_loss) / len(min_bboxes_added_preds_loss)
        orig_classification_loss = sum(orig_classification_loss) / len(orig_classification_loss)

        print(f"average of combined_loss over {len(loader)} batches: {loss}")
        return loss, [max_objects_loss, min_bboxes_added_preds_loss, orig_classification_loss]

    def compute_penalty_term(self, image, init_image):
        return 0

    def max_objects(self, output_patch, conf_thres=0.25, target_class=2):
        #output_patch.size() = torch.Size([8, 25200, 85])
            #8 plaatjes per batch
            #25200 bounding boxes per plaatje
            #85 = (x, y, width, height), object confidence score, and class probabilities. (dus 80 class probs)
        print(output_patch.size())

        #output_patch[:, :, 5:] = alle classes
        #output_patch[:, :, 4:5] = confidence score
        #Je neemt dus een soort expected value van alle classes
        #voor 1 plaatje worden alle 80 class probs vermenigvuldigd met de object confidence score
        x2 = output_patch[:, :, 5:] * output_patch[:, :, 4:5] #torch.Size([8, 25200, 80])     #eq. 2 in paper
        #neem per plaatje, per bb, de max confidence.
        #We houden dus van 80 bbs 1 class over die de hoogste conf heeft, we slaan de conf scores hier op, niet de bbs of de class
        conf, _ = x2.max(2, keepdim=False) #torch.Size([8, 25200])
        #neem van alle plaatjes in de batch (8), van alle bounding boxes per plaatje (25200)..
        #alleen de bounding box met class "car" (dat is index 2 van de 80)
        all_target_conf = x2[:, :, target_class] #torch.Size([8, 25200])
        #neem van alle plaatjes in de batch (8), van alle bounding boxes per plaatje (25200)..
        #conf < conf_thres: "gegeven de confwaarden met ieder behorende tot een specifieke class, 
        # kijk of deze conf waarde lager is dan 0.25". Bijv. een bounding box van een fiets met confwaarde 0.3
        # van alle indices waarbij conf < conf_thres, neem daarvan alle corresponderende "cars" confwaarden
        #dit resulteert in een set van "cars" confwaarden waarbij de hoogste confwaarde van die bounding box (bijv die van een fiets)
        #alsnog lager is dan 0.25, dus de "cars" confwaarde MOET dan ook lager zijn


        #[car, bicycle]
        #[0.2, 0.3] #the highest in this list must be lower than 0.25, so it returns False due to the bicycle
            #this is weird because the confwaarde for car IS below the threshold, why then look at that of bicycle?
            #now we have an idx that says "False" due to some bike having too high of a confwaarde, but we're looking at cars

            #Dit is niet weird, die 0.3 kan betekenen dat er daadwerkelijk een bicycle gezien is in de foto, 
            #dus deze bb wordt genegeerd (False in "conf < conf_thres" mask). Om F2 te 'passeren' moet
            #de max class conf hoger zijn dan 0.25, dat is nu het geval met bicycle = 0.3. 
            #Het doel van deze hele functie is dat de bb niet weggefilterd wordt door F2, dus de hoogste max class conf moet > 0.25
            #ongeacht wat de hoogste class is. 
            #Kortom: We zijn niet geinteresseerd in het verhogen van een confwaarde van een car = 0.2, als de bicycle al 0.3 is.
            #dan is F2 namelijk al gepasseerd; bicycle is namelijk hoog genoeg.
            #indien F2 niet gepasseerd wordt, nemen we de bb van de car, en die proberen we op te krikken

            #the bicycle's 0.3 is too high, so we don't save the car's 0.2
        #[0.3, 0.2] #the highest in this list must be lower than 0.25, so it returns False due to the car being 0.3
            #the car's 0.3 is too high, so we don't save the car's 0.3
        #[0.2, 0.23] #the highest in this list must be lower than 0.25, so it returns True due to the bicycle being 0.23
            #the bicycle's 0.23 is low enough, so we save the car's 0.2

        #print("BB CSs == 0.25", len(all_target_conf[conf == conf_thres]))
        under_thr_target_conf = all_target_conf[conf < conf_thres] #size is rondom torch.Size([200000])

        #print(len(conf.view(-1))) #8 * 25200 = 201600
        #print(len(conf.view(-1)[conf.view(-1) > conf_thres])) #kan verschillen, vaak rond de 500-1000
        #print(len(output_patch)) #8

        #shows the batch average of the number of bbs that go over the conf_thres of 0.25
        #Higher conf_avg, the better. In that case, NMS has more work to do, which is what we want
        conf_avg = len(conf.view(-1)[conf.view(-1) > conf_thres]) / len(output_patch)
        print(f"batch average number (/8) of bbs that will be passed to the NMS stage: {conf_avg}")

        zeros = torch.zeros(under_thr_target_conf.size()).to(output_patch.device) #size is rondom torch.Size([200000])
        zeros.requires_grad = True
        #lager is beter
        #neem 0.25 - confwaarde (confwaarde is altijd lager dan 0.25 omdat alles erboven al False kreeg in de mask)
            #confwaarde van 0.25 en hoger is wenselijk, dan is de loss 0
            #te lage confwaarde zoals 0.10 betekent: 0.25 - 0.10 = 0.15, hogere loss
            #confwaarde moet dus in de buurt van de threshold zitten
            #kortom moedigt de loss functie je aan om te lage confwaarden hoger te maken

            #0.25 - 0.24 = 0.01, klein beetje loss, volgende iteratie zorgt ervoor dat 0.24 dichter bij 0.25 zit
            #0.25 - 0.25 = 0, geen loss, perfect (wel onmogelijk, want de patch regel zet alles onder 0.25 op true, 0.24999999 zou wel kunnen bijv.)
            #0.25 - 0.26 = 0, 0.26 komt hier nooit aan, die is al weggefilterd

            #HET LIJKT DUS ALSOF JE OPTIMALISEERT ZODAT DE CONFWAARDE 0.25 NADERT, MAAR NIET EROVERHEEN GAAT
            #MOET HET, OM F2 TE PASSEREN, NIET HOGER ZIJN DAN 0.25?

            # domein van alle waarden die we omhoog willen hebben zijn [0, 0.25)
            # onze loss functie is max(0.25 - x, 0)
            # de derivative van de loss functie is -1 wanneer x < 0.25
            # -1 is dus onze gradient wanneer conf te laag is 
            # We optimaliseren totdat we een nulpunt bereiken
            # Na een tijd kunenn we een score hebben van 0.24999999
            # Deze wordt dan bijv geoptimaliseerd naar 2.5
            # Dat betekent dat bij de volgende iteratie deze BB CS False wordt; geen optimalisatie meer dus
            # Maar als hij precies 2.5 is, is het nog wel te laag voor de NMS
            # Dit gebeurt in de praktijk echter nooit, hij zit er altijd boven of onder.

            # Om de vraag te beantwoorden, je optimaliseert inderdaad zodanig dat het 0.25 nadert,
            # Maar uit de praktijk lijkt het dat de increase in CS de 0.25 overshoot
            # Dus wanneer de gradient van de loss verwerkt wordt in de UAP,
            # gaat de CS van 0.24999999 naar bijv 0.25000001. 
            # Hierdoor wordt hij eerder in deze functie op False gezet.
            # CS == 0.25 kan in theorie voorkomen, maar gebeurt niet in praktijk

        x3 = torch.maximum(conf_thres - under_thr_target_conf, zeros) #eq. 3 in paper #size is rondom torch.Size([200000])
        #±200000 gedeeld door 201600
        mean_conf = torch.sum(x3, dim=0) / (output_patch.size()[0] * output_patch.size()[1])

        return mean_conf

    def bboxes_area(self, output_clean, output_patch, conf_thres=0.25):

        def xywh2xyxy(x):
            # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
            y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
            y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
            y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
            y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
            y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
            return y

        t_loss = 0.0
        preds_num = 0

        xc_patch = output_patch[..., 4] > conf_thres
        not_nan_count = 0

        # For each img in the batch
        for (xi, x), (li, l) in (zip(enumerate(output_patch), enumerate(output_clean))):  # image index, image inference

            x1 = x[xc_patch[xi]]  # .clone()
            x2 = x1[:, 5:] * x1[:, 4:5]  # x1[:, 5:] *= x1[:, 4:5]

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box_x1 = xywh2xyxy(x1[:, :4])

            min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
            agnostic = True

            conf_x1, j_x1 = x2.max(1, keepdim=True)
            x1_full = torch.cat((box_x1, conf_x1, j_x1.float()), 1)[conf_x1.view(-1) > conf_thres]
            c_x1 = x1_full[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes_x1, scores_x1 = x1_full[:, :4] + c_x1, x1_full[:, 4]  # boxes (offset by class), scores
            final_preds_num = len(torchvision.ops.nms(boxes_x1, scores_x1, conf_thres))
            preds_num += final_preds_num

            # calculate bboxes' area avg
            bboxes_x1_wh = xyxy2xywh(boxes_x1)[:, 2:]
            bboxes_x1_area = bboxes_x1_wh[:, 0] * bboxes_x1_wh[:, 1]
            img_loss = bboxes_x1_area.mean() / (self.patch_size[0] * self.patch_size[1])
            if not torch.isnan(img_loss):
                t_loss += img_loss
                not_nan_count += 1

        if not_nan_count == 0:
            t_loss_f = torch.tensor(torch.nan)
        else:
            t_loss_f = t_loss / not_nan_count

        return t_loss_f

    #calculates the gradient needed for the perturbation
    #wordt elke batch gecalled
    def loss_function_gradient(self, applied_patch, init_images, batch_label, penalty_term, adv_patch):
        #applied_patch = batch + perturbation
        #init_images = images van de batch
        #batch_label = labels van de batch
        #penalty_term = 0
        #adv_patch = zeroes van 3x640x640
        if self.use_cuda:
            init_images = init_images.cuda()
            applied_patch = applied_patch.cuda()
        r = random.randint(0, len(self.models)-1) # choose a random model

        with torch.no_grad():
            #WAT OUTPUT HET MODEL PRECIES?
            output_clean = self.models[r](init_images)[0].detach()
        output_patch = self.models[r](applied_patch)[0]

        max_objects_loss = self.max_objects(output_patch)
        bboxes_area_loss = self.bboxes_area(output_clean, output_patch)
        iou_loss = self.iou(output_clean, output_patch)

        #eerste term uit de loss function: max objects
        loss = max_objects_loss * self.lambda_1
        self.current_max_objects_loss += (self.lambda_1 * max_objects_loss.item())

        #tweede term uit de loss function: bbox area
        if not torch.isnan(bboxes_area_loss):
            loss += (bboxes_area_loss * self.lambda_2)
            self.current_min_bboxes_added_preds_loss += (bboxes_area_loss * self.lambda_2)

        #derde term uit de loss function: IoU loss
        if not torch.isnan(iou_loss):
            #1 - lambda_1 = lambda_3
            loss += (iou_loss * (1 - self.lambda_1))
            self.current_orig_classification_loss += ((1 - self.lambda_1) * iou_loss.item())

        self.current_train_loss += loss.item()
        
        if self.use_cuda:
            loss = loss.cuda()

        print(f"combined_loss, max_obj, bb_area, iou_loss for this batch = \
            {round(loss.item(),3)}|{round(max_objects_loss.item() * self.lambda_1,3)}|{round(bboxes_area_loss.item() * self.lambda_2,3)}|{round(iou_loss.item() * (1 - self.lambda_1),3)}")
        self.models[r].zero_grad()

        #print(type(self.models[r]))
        #print(loss)
        #print(adv_patch)
        #gradient of the loss w.r.t. the adv_patch
        data_grad = torch.autograd.grad(loss, adv_patch)[0]
        #print(f"data grad =  {data_grad[0]}")
        return data_grad

    #FGSM!
    def fastGradientSignMethod(self, adv_patch, images, labels, epsilon=0.3):

        #alle plaatjes in de batch met de perturbation toegevoegd
        applied_patch = torch.clamp(images[:] + adv_patch, 0, 1)
        #Is altijd 0
        penalty_term = self.compute_penalty_term(images, images)
        #applied_patch = batch + perturbation
        #images = images van de batch
        #labels = labels van de batch
        #penatly_term = 0
        #adv_patch = zeroes van 3x640x640
        data_grad = self.loss_function_gradient(applied_patch, images, labels, penalty_term,
                                                adv_patch) 
        
        # Collect the element-wise sign of the data gradient
        #returnt -1, 0 of 1
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        # we doen "- epsilon * sign_data_grad"
        # "-" betekent dat we een perturbed_patch krijgen waarvan de loss lager zou moeten zijn
        #Als we "+" zouden doen, zouden we ascenden in de loss landscape, en wordt de loss dus hoger, terwijl we die laag willen hebben
        perturbed_patch = adv_patch - epsilon * sign_data_grad
        #print(f"perturbed patch: {perturbed_patch[0]}")
        
        # Adding clipping to maintain [0,1] range
        perturbed_patch_c = torch.clamp(perturbed_patch, 0, 1).detach()
        #print(f"clamped perturbed patch: {perturbed_patch_c[0]}")
        # Return the perturbed image
        return perturbed_patch_c

    def pgd_L2(self, epsilon=0.1, iter_eps=0.05, min_x=0.0, max_x=1.0):
        early_stop = EarlyStopping(delta=1e-4, current_dir=self.current_dir, patience=7)

        #patch size is 640x640, gewoon ter grootte van een normaal plaatje
        patch_size = self.patch_size
        #maakt een tensor van 3x640x640 gevuld met nullen
        patch = torch.zeros([3, patch_size[0], patch_size[1]])
        #zorgt ervoor dat je de gradient kan nemen dmv x.grad bijv, of je kan backprop doen dmv y.backward()
        #ik geloof niet dat we dit nodig hebben aangezien we niet echt backpropagation doen
        patch.requires_grad = True

        adv_patch = patch
        #1 epoch == alle batches 1 keer langs laten gaan
        for epoch in range(self.epochs):
            #epoch_length = 169, aantal batches per epoch (169 * 8 = 1352, bijna 1350 zoals de bedoeling is, want val is 150 en test is 500)
            epoch_length = len(self.train_loader)
            print('Epoch: ', epoch)
            # if epoch == 0:
            #     #leg de allereerste val loss vast voor early stopping comparison
            #     #de IoU loss hoort 0 te zijn:
            #     #self.iou neemt de output van het model wanneer img_batch wordt gepasst en de output wanneer applied_batch wordt gepasst
            #     #"output_clean = self.models[r](img_batch)[0]"
            #     #"output_patch = self.models[r](applied_batch)[0]"
            #     #
            #     #"iou = self.iou(output_clean, output_patch)"
            #     #output_clean en output_patch verschillen niks van elkaar, 
            #     #aangezien de perturbations bij epoch 0 = torch.zeros([3, patch_size[0], patch_size[1]])
            #     #Het is dan logisch dat de loss daartussen 0 is, de inputs vor de iou loss functie zijn identiek
            #     val_loss = self.evaluate_loss(self.val_loader, adv_patch)[0]
            #     early_stop(val_loss, adv_patch.cpu(), epoch)

            # Perturb the input
            self.current_train_loss = 0.0
            self.current_max_objects_loss = 0.0
            self.current_min_bboxes_added_preds_loss = 0.0
            self.current_orig_classification_loss = 0.0

            i = 0
            #Loop door alle batches heen, elke batch bevat 8 imgs, alle labels per image, en de paden naar de images (maar die skippen we)
            for (imgs, label, _) in self.train_loader:  # for imgs, label in self.train_loader:#self.coco_train:
                if i % 5 == 0:
                    print(f"saving png of current patch at epoch: {epoch}, batch: {i}/{len(self.train_loader)}...")
                    #per 25 batches, sla de adv patch van dat moment op als png
                    patch_n = self.full_patch_folder + f"uap_epoch={epoch}_batch={i}.png"
                    transp(adv_patch).save(patch_n)

                #Maakt van een tuple van 8 images een torch van 8x3x640x640
                x = torch.stack(imgs)

                #returnt middels FGSM een perturbed plaatje van 3x640x640
                #adv_patch = bij epoch 0, batch 0 is dit een zwart plaatje (tensor met 0's)
                #x = één tensor batch van 8 plaatjes
                #label = een tuple van 8, elke entry bevat een np.array met elke row een bb, en columns [class, x-center, y-center, w, h]
                #iter_eps = 0.0005, bepaalt de sterkte van de perturbation
                adv_patch = self.fastGradientSignMethod(adv_patch, x, label, epsilon=iter_eps)
                #torch.save(adv_patch, f'self.full_patch_folder + tensor_{i}.pt')
                #print(f"adv_patch: {adv_patch}")s

                # Project the perturbation to the epsilon ball (L2 projection)
                # Neem het verschil tussen het zwarte plaatje en het perturbed zwarte plaatje
                perturbation = adv_patch - patch

                #calculates the L2-norm
                #√pert_idx1^2 + pert_idx2^2 + pert_idx3^2
                norm = torch.sum(torch.square(perturbation))
                norm = torch.sqrt(norm)

                #Here we constrain the perturbed x such that its perturbation does not exceed epsilon
                #x = clean x
                #x' = perturbed x
                #x − x′ = perturbation (hierboven berekend (norm))
                #S = {x′: ∥x − x′∥p < ϵ}.
                
                #HOEZO STAAT DIT ERTUSSEN? 
                #factor kan niet groter zijn dan 1
                #epsilon = 0.1
                #min(1, 0.1/0.1) = 1
                #min(1, 0.1/0.05) = 1
                #min(1, 0.1/0.5) = 0.5
                #Een te hoge norm leidt tot een kleinere factor,
                #Een te lage norm leidt tot factor 1
                #Als dus de perturbation extreem erg afwijkt van de clean patch (hoge norm) dan wordt pert afgeremd middels de factor
                factor = min(1, epsilon / norm.item())  # torch.divide(epsilon, norm.numpy()[0]))

                #min_x = 0, max_x = 1
                #maak de perturbed image (x'), waarbij de perturbation * factor is, 
                #en de range loopt van 0 tot 1
                adv_patch = (torch.clip(patch + perturbation * factor, min_x, max_x))  # .detach()

                i += 1
                #wanneer i == 169
                if i == epoch_length:
                    #Doet dus in principe de eindresultaten van een epoch vastleggen, oa de validation loss
                    self.last_batch_calc(adv_patch, epoch_length, epoch, i)

            # check if loss has decreased
            #als de loss niet decreast voor 7 epochs, kap de boel af en ga naar return
            if early_stop(self.val_losses[-1], adv_patch.cpu(), epoch):
                self.final_epoch_count = epoch
                break

        print("Training finished")
        #is gewoon 3x640x640 adversarial patch
        return early_stop.best_patch

    def run_attack(self):
        tensor_adv_patch = self.pgd_L2(epsilon=self.epsilon, iter_eps=0.0005) 

        patch = tensor_adv_patch

        #schiet een plaatje van de final adv patch
        #slaat ook de train/val losses etc per epoch op en stopt ieder in een lijst
        self.save_final_objects(tensor_adv_patch)
        #maakt er een zichtbaar plaatje van
        adv_image = transp(patch[0])

        return adv_image

