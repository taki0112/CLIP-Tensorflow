import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
import tensorflow.keras.layers as nn
from tensorflow import einsum


from contextlib import contextmanager
from functools import partial
from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange

import numpy as np

# helper functions
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

@contextmanager
def null_context():
    yield

def max_neg_value(dtype):
    return -np.finfo(dtype.as_numpy_dtype).max

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

def masked_mean(t, mask, dim = 1, eps = 1e-6):
    t = masked_fill(t, ~mask, 0.0)
    numer = tf.reduce_sum(t, axis=dim)

    denorm = tf.reduce_sum(mask, axis=dim)
    denorm = tf.clip_by_value(denorm, clip_value_min=eps, clip_value_max=tf.reduce_max(denorm))

    return numer / denorm

def log(t, eps = 1e-20):
    return tf.math.log(t + eps)

def l2norm(t):
    return tf.math.l2_normalize(t, axis=-1)

def masked_select(x, mask):
    x = tf.cast(x, tf.float32)
    mask = tf.cast(mask, tf.int32)

    x = tf.reshape(x, [-1])
    mask = tf.reshape(mask, [-1])
    mask_true_idx = tf.where(mask)

    return tf.gather_nd(x, mask_true_idx)

def masked_fill(x, mask, true_val):
    x = tf.where(mask, true_val, x)
    return x

def matrix_diag(t):
    # t.shape = [1,4,4]
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = tf.range(i)
    j_range = tf.range(j)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j') # [4,4]

    diag_el = masked_select(t, diag_mask) # [4]

    return rearrange(diag_el, '(b d) -> b d', d = num_diag_el)

# keyword argument helpers
def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    return str.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

# helper classes
class LayerNorm(Layer):
    # bias-less layernorm
    def __init__(self, dim, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.g = tf.Variable(tf.ones([dim]))

    def call(self, x, training=True):
        var = tf.math.reduce_variance(x, axis=-1, keepdims=True)
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)

        x = (x - mean) / tf.sqrt((var + self.eps)) * self.g
        return x

class PreNorm(Layer):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()

        self.norm = LayerNorm(dim)
        self.fn = fn

    def call(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# rotary positional embedding
class RotaryEmbedding(Layer):
    def __init__(self, dim):
        super(RotaryEmbedding, self).__init__()
        self.inv_freq = 1.0 / (10000 ** (tf.range(0, dim, 2, dtype=tf.float32) / dim))

    def call(self, seq_len, training=True):
        inv_freq = self.inv_freq
        t = tf.cast(tf.range(seq_len), dtype=inv_freq.dtype)
        freqs = einsum('i , j -> i j', t, inv_freq)

        x = tf.concat([freqs, freqs], axis=-1)
        return x

def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = tf.unstack(x, axis=-2)

    x = tf.concat([-x2, x1], axis=-1)
    return x

def apply_rotary_pos_emb(freqs, t):
    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * tf.math.cos(freqs)) + (rotate_half(t) * tf.math.sin(freqs))

    x = tf.concat([t, t_pass], axis=-1)
    return x

# transformer
class SwiGLU(Layer):
    def __init__(self):
        super(SwiGLU, self).__init__()

    def silu(self, x):
        return x * tf.sigmoid(x)

    def call(self, x, training=True):
        x, gates = tf.split(x, num_or_size_splits=2, axis=-1)
        return x * self.silu(gates)

class MLP(Layer):
    def __init__(self, dim, mult=4, dropout=0.0):
        super(MLP, self).__init__()
        inner_dim = int(dim * mult)


        self.net = Sequential([
            nn.Dense(units=inner_dim * 2, use_bias=False),
            SwiGLU(),
            nn.Dropout(rate=dropout),
            nn.Dense(units=dim, use_bias=False)
        ])

    def call(self, x, training=True):
        return self.net(x, training=training)


class Attention(Layer):
    def __init__(self, dim, dim_head=64, heads=8, dropout=0.0):
        super(Attention, self).__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads


        self.to_qkv = nn.Dense(units=inner_dim * 3, use_bias=False)
        self.to_out = nn.Dense(units=dim, use_bias=False)
        self.dropout = nn.Dropout(rate=dropout)

    def call(self, x, mask=None, rotary_pos_emb=None, training=True):
        h = self.heads
        qkv = self.to_qkv(x)
        qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        if exists(rotary_pos_emb):
            apply_rotary = partial(apply_rotary_pos_emb, rotary_pos_emb)
            q, k, v = map(apply_rotary, (q, k, v))

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            mask_value = max_neg_value(sim.dtype)
            sim = masked_fill(sim, ~mask, mask_value)

        attn = tf.nn.softmax(sim, axis=-1)
        attn = self.dropout(attn, training=training)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out

class Transformer(Layer):
    def __init__(self, dim, depth, dim_head=64, heads=8, attn_dropout=0.0, ff_mult=4):
        super(Transformer, self).__init__()

        self.layers = []

        for _ in range(depth):
            self.layers.append([
                PreNorm(dim, Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)),
                PreNorm(dim, MLP(dim=dim, mult=ff_mult))
            ])

        self.norm_out = LayerNorm(dim)

    def call(self, x, rotary_pos_emb=None, mask=None, training=True):
        for attn, ff in self.layers:
            x = attn(x, mask=mask, rotary_pos_emb=rotary_pos_emb, training=training) + x
            x = ff(x, training=training) + x

        x = self.norm_out(x)
        return x

# text and vision transformers
class TextTransformer(Layer):
    def __init__(self, dim, num_tokens, max_seq_len, dim_head, rotary_pos_emb=None, **kwargs):
        super(TextTransformer, self).__init__()

        self.token_emb = nn.Embedding(num_tokens, dim)

        self.abs_pos_emb = nn.Embedding(max_seq_len, dim) if not rotary_pos_emb else None
        self.rotary_pos_emb = RotaryEmbedding(min(dim_head, 32)) if rotary_pos_emb else None

        self.cls_token = tf.Variable(tf.random.normal(shape=[dim]))

        self.transformer = Transformer(dim, dim_head=dim_head, **kwargs)

    def call(self, x, mask=None, training=True):
        b, n = x.shape

        x = self.token_emb(x)

        if exists(self.abs_pos_emb):
            pos_emb = self.abs_pos_emb(tf.range(n))
            x = x + rearrange(pos_emb, 'n d -> 1 n d')

        rotary_pos_emb = None
        if exists(self.rotary_pos_emb):
            rotary_pos_emb = self.rotary_pos_emb(n + 1)

        cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b=b)
        x = tf.concat([cls_tokens, x], axis=1)

        if exists(mask):
            mask = tf.pad(mask, paddings=[[0,0], [1,0]], constant_values=True)

        out = self.transformer(x, mask=mask, rotary_pos_emb=rotary_pos_emb, training=training)
        return out

class VisionTransformer(Layer):
    def __init__(self, dim, image_size, patch_size, **kwargs):
        super(VisionTransformer, self).__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2

        self.cls_token = tf.Variable(tf.random.normal(shape=[dim]))

        self.to_tokens = Sequential([
            Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Dense(units=dim)
        ])

        self.pos_emb = nn.Embedding(num_patches, dim)
        self.transformer = Transformer(dim, **kwargs)

    def call(self, x, training=True):

        x = self.to_tokens(x)
        b, n, _ = x.shape

        pos_emb = self.pos_emb(tf.range(n))
        x = x + rearrange(pos_emb, 'n d -> 1 n d')

        cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b=b)
        x = tf.concat([cls_tokens, x], axis=1)

        out = self.transformer(x, training=training)
        return out

# contrastive learning functions
def model_forward_with_context(fn, args, freeze):
    enc = fn(*args)
    if freeze:
        enc = tf.stop_gradient(enc)

    return enc

# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits) :
    return tf.math.reduce_mean(
        tf.keras.metrics.sparse_categorical_crossentropy(
            y_true=tf.range(logits.shape[0]), y_pred=logits, from_logits=True
        )
    )

def clip_loss(text_embeds, image_embeds, logit_scale) :
    # normalized features
    image_embeds = image_embeds / tf.norm(tensor=image_embeds, ord="euclidean", axis=-1, keepdims=True)
    text_embeds = text_embeds / tf.norm(tensor=text_embeds, ord="euclidean", axis=-1, keepdims=True)

    # cosine similarity as logits
    logit_scale = tf.math.exp(logit_scale)
    logits_per_text = tf.matmul(text_embeds, image_embeds, transpose_b=True) * logit_scale
    logits_per_image = tf.transpose(logits_per_text)
    similarity = logits_per_text

    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(tf.transpose(similarity))
    return (caption_loss + image_loss) / 2.0

# https://github.com/lucidrains/x-clip
def lucidrains_loss(text_latents, image_latents, temperature):
    # equal to clip_loss
    num_batch_texts = num_batch_images = 1
    text_latents, image_latents = map(l2norm, (text_latents, image_latents))

    # get temperature
    temp = tf.exp(temperature)

    # split out multiview dimension for text and images
    text_latents = rearrange(text_latents, '(m b) ... -> m b ...', m=num_batch_texts)
    image_latents = rearrange(image_latents, '(m b) ... -> m b ...', m=num_batch_images)

    # calculate loss
    text_to_image = einsum('m t d, n i d -> m n t i', text_latents, image_latents) * temp
    image_to_text = rearrange(text_to_image, '... t i -> ... i t')

    text_to_image = rearrange(text_to_image, 'm n ... -> (m n) ...')
    image_to_text = rearrange(image_to_text, 'm n ... -> (m n) ...')

    # exponentiate
    text_to_image_exp, image_to_text_exp = map(tf.exp, (text_to_image, image_to_text))

    # numerators
    text_to_image_pos, image_to_text_pos = map(matrix_diag, (text_to_image_exp, image_to_text_exp))

    # denominator
    text_to_image_denom, image_to_text_denom = map(lambda t: tf.reduce_sum(t, axis=-1),
                                                   (text_to_image_exp, image_to_text_exp))

    # loss
    text_to_image_loss = tf.reduce_mean(-log(text_to_image_pos / text_to_image_denom), axis=-1)
    image_to_text_loss = tf.reduce_mean(-log(image_to_text_pos / image_to_text_denom), axis=-1)

    # calculate CL loss
    cl_loss = (text_to_image_loss + image_to_text_loss) / 2

    return cl_loss

# main clip class
class CLIP(Model):
    def __init__(self,
                image_encoder=None,
                text_encoder=None,
                dim_text=512,
                dim_image=512,
                dim_latent=512,
                num_text_tokens=10000,
                text_enc_depth=6,
                text_seq_len=256,
                text_heads=8,
                text_dim_head=64,
                text_has_cls_token=True,
                text_pad_id=0,
                text_rotary_pos_emb=False,
                visual_enc_depth=6,
                visual_heads=8,
                visual_dim_head=64,
                visual_image_size=256,
                visual_patch_size=32,
                visual_has_cls_token=True
                ):
        super(CLIP, self).__init__()
        assert (visual_has_cls_token or text_has_cls_token), 'CLS token must be included on both vision and text transformers if you are not using fine-grained contrastive learning loss'
        # store some parameters for access
        self.dim_text = dim_text
        self.dim_image = dim_image
        self.dim_latent = dim_latent

        # instantiate text transformer
        self.text_pad_id = text_pad_id
        self.text_has_cls_token = text_has_cls_token

        if exists(text_encoder):
            self.text_transformer = text_encoder
        else:
            self.text_transformer = TextTransformer(
                dim=dim_text,
                num_tokens=num_text_tokens,
                max_seq_len=text_seq_len,
                depth=text_enc_depth,
                heads=text_heads,
                dim_head=text_dim_head,
                rotary_pos_emb=text_rotary_pos_emb
            )

        # instantiate image transformer
        self.visual_has_cls_token = visual_has_cls_token

        if exists(image_encoder):
            self.visual_transformer = image_encoder
        else:
            self.visual_transformer = VisionTransformer(
                dim=dim_image,
                image_size=visual_image_size,
                patch_size=visual_patch_size,
                depth=visual_enc_depth,
                heads=visual_heads,
                dim_head=visual_dim_head
            )

        # text latent projection
        self.to_text_latent = nn.Dense(units=dim_latent, use_bias=False)

        # image latent projection
        self.to_visual_latent = nn.Dense(units=dim_latent, use_bias=False)

        # temperature
        self.temperature = tf.Variable(tf.constant(1.0, dtype=tf.float32))

    def call(self, text, image=None, training=True,
             return_loss=False,
             return_encodings=False,
             freeze_image_encoder=False,  # image encoder is not trained if this is set to True, proposed by LiT paper
             freeze_text_encoder=False,  # text encoder is not trained if this is set to True
             **kwargs
             ):

        # derive text mask
        text_mask = text != self.text_pad_id


        assert not (return_loss and not training), 'loss cannot be used if not training'

        # get encoded text
        enc_text = model_forward_with_context(
            fn=self.text_transformer,
            args=(text, text_mask, training),
            freeze=freeze_text_encoder
        )

        # whether to train image encoder, in the case that the image net was pretrained as recommended in LiT
        enc_image = model_forward_with_context(
            fn=self.visual_transformer,
            args=(image, training),
            freeze=freeze_image_encoder
        )

        # early return of encodings, if needed (for DALL-E2)
        if return_encodings:
            return enc_text, enc_image

        # depending on whether to do fine-grained CLIP or not, select either all tokens, or CLS tokens only
        text_embeds = enc_text[:, 0]
        image_embeds = enc_image[:, 0]

        # project to latents
        text_latents = self.to_text_latent(text_embeds)
        image_latents = self.to_visual_latent(image_embeds)

        # calculate loss
        # cl_loss = lucidrains_loss(text_latents, image_latents, self.temperature)
        cl_loss = clip_loss(text_latents, image_latents, self.temperature)

        # calculate weights
        cl_loss_weight = 1

        loss = cl_loss * cl_loss_weight

        return loss


