from sian.models.models import (
    MLP,
    SIAN,
    FeedForwardDNN
)

# from sian.models.smooth_sian_or_mnist_sian import (
#     Smooth_Or_MNIST_SIAN,
# )




from sian.models.training import (
    TrainingArgs,
    # normal_gradient_descent_training,
    # masked_gradient_descent_training,
    either_normal_or_masked___gradient_descent_training,
    evaluate_model_on_test_set,
)

from sian.models.masked_models import (
    MaskedMLP,
    InstaSHAPMasked_SIAN,
    # MaskedGAM, #outdated version?
)

# from .jun22_gamgood_models import mnist_Adaptive_DNN_plus_GAM as mnist_SIAN



# from .smooth_sian_or_mnist_sian import Smooth_Or_MNIST_SIAN
# from .smooth_sian_or_mnist_sian import SmoothOrMnist_InstaSHAPMasked_SIAN
from .new_SoM_models import SmoothOrMnist_UnmaskedOrMasked_SIAN
from .new_SoM_models import SmoothOrMnist_UnmaskedOrMasked_MLP


from .new_inflated_models import SemiInflated_SmoothOrMnist_UnmaskedOrMasked_SIAN



from .vision_models import Mnist_CnnModel
from .vision_models import Mnist_CnnGam1x1, Mnist_CnnGam2x2
# from .vision_models import Mnist_CnnGam2x2_plusLongRange


