from copy import deepcopy
from functools import partial
import torch
import torch.nn as nn


def _init_vit_weights(m):
    """
    PiT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Pillar_Encoder(nn.Module):
    def __init__(self, embed_dim=768, interval=0.2, layer_setting=([6, 32], [32, 64], [64, 128]), aggregation_setting=([256, 768],), norm_layer=None,
                 pie_drop_ratio=0.5):
        super(Pillar_Encoder, self).__init__()

        self.embed_dim = embed_dim
        self.interval = interval
        self.drop = nn.Dropout(pie_drop_ratio)
        self.sequential1_linear_bn_RL = nn.Sequential()
        self.sequential2_linear_bn_RL = nn.Sequential()
        self.bn = nn.BatchNorm1d(100)
        for in_feature, out_feature in layer_setting:
            linear = nn.Linear(in_features=in_feature, out_features=out_feature, bias=False)
            # bn = nn.BatchNorm1d(num_features=out_feature)
            RL = nn.ReLU()
            # drop = nn.Dropout(pie_drop_ratio)
            self.sequential1_linear_bn_RL.append(linear)
            # self.sequential1_linear_bn_RL.append(bn)
            self.sequential1_linear_bn_RL.append(RL)
            # self.sequential1_linear_bn_RL.append(drop)
        for in_feature, out_feature in aggregation_setting:
            linear = nn.Linear(in_features=in_feature, out_features=out_feature, bias=False)
            # bn = nn.BatchNorm1d(num_features=out_feature)
            RL = nn.ReLU()
            # drop = nn.Dropout(pie_drop_ratio)
            self.sequential2_linear_bn_RL.append(linear)
            # self.sequential2_linear_bn_RL.append(bn)
            self.sequential2_linear_bn_RL.append(RL)
            # self.sequential2_linear_bn_RL.append(drop)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def pillar_allocation(self, one_data: torch.tensor):
        """
           input: tensor，单个样本的所有点，第一列数据为代表Y轴上坐标，第二列数据为代表Z轴上坐标，第三列数据为代表X轴上坐标
           output:
                groups:  不同体素中所有点的集合
                index2indices: groups中每个集合对应的体素坐标
        """
        one_data = one_data.cpu()
        indices2index = {}
        index2indices = {}
        groups = list()
        temp_x, temp_y, temp_z = deepcopy(one_data[:, 2]), deepcopy(one_data[:, 0]), deepcopy(one_data[:, 1])
        one_data[:, 0], one_data[:, 1], one_data[:, 2] = temp_z, temp_y, temp_x
        num_of_point = one_data.shape[0]
        for i in range(num_of_point):
            indices = tuple((torch.div((one_data[i][1:] + 1.0).clamp(0, 1.99), self.interval, rounding_mode="trunc")).numpy().tolist())
            if indices in indices2index.keys():
                index = indices2index[indices]
                groups[index].append(one_data[i])
            else:
                indices2index[indices] = len(groups)
                groups.append([])
                groups[-1].append(one_data[i])
        for indices, index in indices2index.items():
            index2indices[index] = indices

        return index2indices, groups

    def augmented_feature(self, single_pillar):
        """
            input: tensor，单个柱形中的所有点 [point_num, 3]
            output:
                augmented_data:  依据质心来增强单个体素中所有点
        """
        point_collection = torch.stack(single_pillar, 0)
        centroid = torch.unsqueeze(torch.mean(point_collection, 0), 0)
        point2centroid = point_collection - centroid
        augmented_data = torch.cat([point_collection, point2centroid], 1)
        # augmented_data: [point_num, 6]
        return augmented_data

    def forward(self, x: torch.tensor):
        # x: [batch_size, point_num, 3]
        sparse_matrix_collection = list()
        for i in range(x.shape[0]):
            index2indices, groups = self.pillar_allocation(x[i])
            sparse_matrix = torch.zeros(size=(self.embed_dim, int(2 / self.interval), int(2 / self.interval))).cuda()
            for index, group in enumerate(groups):
                augmented_feature = self.augmented_feature(group)
                step_1 = self.sequential1_linear_bn_RL(augmented_feature.cuda())
                overall_feature = torch.unsqueeze(torch.max(step_1, 0).values, 0)
                copy_overall_feature = overall_feature.repeat(step_1.shape[0], 1)
                step_2 = self.sequential2_linear_bn_RL(torch.cat([step_1, copy_overall_feature], 1))
                one_pillar_encoding = torch.max(step_2, 0)
                indices = index2indices[index]
                y_loc, x_loc = int(indices[0]), int(indices[1])
                sparse_matrix[:, y_loc, x_loc] = one_pillar_encoding.values
            sparse_matrix_collection.append(sparse_matrix)
        # batch_sparse_matrix: [batch_size, dim, 2 / self.interval, 2 / self.interval]
        batch_sparse_matrix = torch.stack(sparse_matrix_collection, 0)
        # batch_sparse_sequences: [batch_size, 2 / self.interval * 2 / self.interval, dim]
        batch_sparse_sequences = self.bn(batch_sparse_matrix.flatten(2).permute(0, 2, 1))
        return batch_sparse_sequences


class Position_Embedding(nn.Module):
    def __init__(self, choice="1D", interval=0.2, embed_dim=768, pos_embed_drop_ratio=0., ):
        super(Position_Embedding, self).__init__()
        self.choice = choice
        self.nums = int(2 / interval)
        self.num_patches = self.nums ** 2
        if choice == "1D":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
            # Weight init
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif choice == "2D":
            self.pos_embed_c = nn.init.trunc_normal_(nn.Parameter(torch.zeros(1, embed_dim)), std=0.02)
            self.pos_embed_x = nn.init.trunc_normal_(nn.Parameter(torch.zeros(self.nums, embed_dim // 2)), std=0.02)
            self.pos_embed_y = nn.init.trunc_normal_(nn.Parameter(torch.zeros(self.nums, embed_dim // 2)), std=0.02)

        self.pos_drop = nn.Dropout(p=pos_embed_drop_ratio)

    def forward(self, x):
        # x: [B, 101, 768]
        if self.choice == "2D":
            x[:, 0, :] += self.pos_embed_c
            i = 1
            for each_y in self.pos_embed_y:
                for each_x in self.pos_embed_x:
                    x[:, i, :] += torch.unsqueeze(torch.cat((each_x, each_y)), dim=0)
                    i += 1
            return self.pos_drop(x)
        elif self.choice == "1D":
            return self.pos_drop(x + self.pos_embed)
        else:
            return x


class Attention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, qkv_bias=False, qk_scale=None, atten_drop_ratio=0., proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(atten_drop_ratio)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed + dim]
        B, N, C = x.shape

        # qkv() -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # q, k, v: [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # transpose -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @ -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @ -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, mlp_drop_ratio=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(mlp_drop_ratio)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, proj_drop_ratio=0.,
                 mlp_drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(embed_dim)
        self.attn = Attention(embed_dim, num_heads, qkv_bias, qk_scale, attn_drop_ratio, proj_drop_ratio)

        # Note: drop path for stochastic depth, we shall see if this is better than dropout here, temporary unused
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(in_features=embed_dim, hidden_features=mlp_hidden_dim, out_features=embed_dim, act_layer=act_layer, mlp_drop_ratio=mlp_drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Pillar_Transformer(nn.Module):
    def __init__(self, interval=0.2, num_classes=40, embed_dim=768, layer_setting=([6, 32], [32, 64], [64, 128]),
                 aggregation_setting=([256, 768],), choice="1D", depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, pie_drop_ratio=0., pos_embed_drop_ratio=0., proj_drop_ratio=0., attn_drop_ratio=0., mlp_drop_ratio=0.,
                 drop_path_ratio=0.2, norm_layer=None, act_layer=None):
        super(Pillar_Transformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        # norm_layer: nn.LayerNorm, act_layer: nn.GELU
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.pillar_encoder = Pillar_Encoder(embed_dim=embed_dim, interval=interval, layer_setting=layer_setting,
                                             aggregation_setting=aggregation_setting, norm_layer=None, pie_drop_ratio=pie_drop_ratio)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_embed = Position_Embedding(choice=choice, interval=interval, embed_dim=embed_dim,
                                            pos_embed_drop_ratio=pos_embed_drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  proj_drop_ratio=proj_drop_ratio, mlp_drop_ratio=mlp_drop_ratio, attn_drop_ratio=attn_drop_ratio,
                  drop_path_ratio=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Classifier head(s)
        self.classifier_head = nn.Linear(self.num_features, num_classes)

        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # [batch_size, point_num, 3] -> [batch_size, num_patches, embed_dim]
        x = self.pillar_encoder(x)  # [B, 100, 768]
        # [1, 1, 768] -> [B, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat((cls_token, x), dim=1)  # [B, 101, 768]
        x = self.pos_embed(x)
        x = self.blocks(x)
        x = self.norm(x)

        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.classifier_head(x)
        return x


def pit_base_patch_point1(num_classes: int = 40):
    drop_ratio = 0.5
    model = Pillar_Transformer(interval=0.2, num_classes=num_classes, embed_dim=768, layer_setting=([6, 32], [32, 64], [64, 128]),
                               aggregation_setting=([256, 768],), choice="1D", depth=2, num_heads=12, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                               pie_drop_ratio=drop_ratio, pos_embed_drop_ratio=drop_ratio, proj_drop_ratio=drop_ratio, attn_drop_ratio=drop_ratio,
                               mlp_drop_ratio=drop_ratio, drop_path_ratio=drop_ratio, norm_layer=None, act_layer=None)

    return model
