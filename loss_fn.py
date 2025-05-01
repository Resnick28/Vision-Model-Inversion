import torch.nn.functional as F
import torch

def cosine_similarity_loss(features):
    normalized_features = F.normalize(features, p=2, dim=1)
    similarity_matrix = torch.mm(normalized_features, normalized_features.t())
    mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device).bool()
    similarity_matrix = similarity_matrix.masked_fill(mask, 0)
    loss = similarity_matrix.sum() / (features.size(0) * (features.size(0) - 1))
    return loss  # Minimize

def feature_orthogonality_loss(features):
    gram_matrix = torch.mm(features, features.t())
    identity_matrix = torch.eye(gram_matrix.size(0), device=gram_matrix.device)
    loss = torch.mean((gram_matrix - identity_matrix) ** 2)
    return loss / (features.size(0) * features.size(1))  # Minimize

def kl_divergence(p, q):

    kl_div = F.kl_div(F.log_softmax(p, dim=1), q, reduction='batchmean')
    return kl_div

def total_variation_loss(generated_images):

    batch_size = generated_images.size(0)
    h_variation = torch.pow(generated_images[:, :, 1:, :] - generated_images[:, :, :-1, :], 2).sum()
    w_variation = torch.pow(generated_images[:, :, :, 1:] - generated_images[:, :, :, :-1], 2).sum()

    return (h_variation + w_variation) / batch_size

def pixel_range_loss(gen_images):
    # Penalize pixels that are below 0 or above 1
    lower_bound_loss = torch.sum(torch.clamp(-gen_images - 1, min=0))  # Penalize pixels less than -1
    upper_bound_loss = torch.sum(torch.clamp(gen_images - 1, min=0))  # Penalize pixels greater than 1

    # Total loss is the sum of both penalties
    total_loss = lower_bound_loss + upper_bound_loss
    return total_loss

def gradient_minimization_loss(model, random_images, target_labels, criterion):

    # Forward pass: Get model output for random images
    model_output = model(random_images)

    # Compute the loss for the random images
    loss_random = criterion(model_output, target_labels)

    # Compute the gradients of the loss with respect to the model's parameters (weights and biases)
    grads = torch.autograd.grad(outputs=loss_random, inputs=model.parameters(),
                                grad_outputs=torch.ones_like(loss_random),
                                create_graph=True)  # Create graph for higher-order derivatives

    # Compute the L2 norm of the gradients
    grad_norm = torch.sqrt(sum([grad.norm(2)**2 for grad in grads]))

    return grad_norm