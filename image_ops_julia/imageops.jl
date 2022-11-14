using Images, ImageDraw
using Colors
using StaticLint
using Test


draw!
######################################################################################################
######################################################################################################
######################################################################################################

function sobelx(img_grey::Matrix{Gray{Float32}})::Matrix{Gray{Float32}}
    height, width = size(img_grey)
    img2 = zeros(Gray{Float32}, height, width)
    for j in 2:(width-1)
        for i in 2:(height-1)
            img2[i, j] = img_grey[i-1, j-1] + 2 * img_grey[i-1, j] + img_grey[i-1, j+1] +
                         -img_grey[i+1, j-1] - 2 * img_grey[i+1, j] - img_grey[i+1, j+1]
        end
    end
    return img2
end

function sobely(img_grey::Matrix{Gray{Float32}})::Matrix{Gray{Float32}}
    height, width = size(img_grey)
    img2 = zeros(Gray{Float32}, height, width)
    for j in 2:(width-1)
        for i in 2:(height-1)
            img2[i, j] = img_grey[i-1, j-1] + 2 * img_grey[i, j-1] + img_grey[i+1, j-1] +
                         -img_grey[i-1, j+1] - 2 * img_grey[i, j+1] - img_grey[i+1, j+1]
        end
    end
    return img2
end

function sobel(img_grey::Matrix{Gray{Float32}})::Matrix{Gray{Float32}}
    gx = sobelx(img_grey)
    gy = sobely(img_grey)
    return sqrt.((gx .^ 2) .+ (gy .^ 2))
end

######################################################################################################
######################################################################################################
######################################################################################################

function hugh_transform_accumulator_matrix(edge_img::Matrix{Gray{Float32}}, resolution::Tuple{Int,Int}=(32, 32), threshhold::Float32=Float32(0.8))
    height, width = size(edge_img)
    ϕ_segements, r_segments = resolution
    
    ϕ_sins::Array{Float32} = map(1:ϕ_segements) do i
        return sin(map_index_to_phi(i, ϕ_segements))
    end
    
    ϕ_coss::Array{Float32} = map(1:ϕ_segements) do i
        return cos(map_index_to_phi(i, ϕ_segements))
    end

    # fill accumulator matrix
    acc_matrix = zeros(Int, resolution)
    maximum::Int = 0
    wh = (width, height)
    for x in 1:width
        for y in 1:height
            val = edge_img[y, x]
            if val > threshhold
                x_rel, y_rel = abs_pos_to_rel((x, y), wh)
                for ϕ_index in 1:ϕ_segements
                    r = x_rel * ϕ_coss[ϕ_index] + y_rel * ϕ_sins[ϕ_index]
                    r_index = map_r_to_index(r, r_segments, wh)
                    acc_matrix[ϕ_index, r_index] += 1
                    if acc_matrix[ϕ_index, r_index] > maximum
                        maximum = acc_matrix[ϕ_index, r_index]
                    end
                end
            end
        end
    end
    # normalize the acc_matrix to float values:
    normalized_acc_matrix = map(acc_matrix) do e
        Float32(e) / Float32(maximum)
    end

    return normalized_acc_matrix
end



# tuples represent: phi_index, r_index, value
function top_k_from_acc_matrix(acc_matrix::Array{Float32}, k::Int)::Array{Tuple{Int,Int,Float32}}
    arr = [(Tuple(ind)[1], Tuple(ind)[2], val) for (ind, val) in pairs(acc_matrix)]
    list = []
    for (y, x, val) in arr
        push!(list, (y, x, val))
    end
    sort!(list, by=x -> x[3])
    top_k = list[length(list)-k+1:end]
    return top_k
end

function top_k_line_params_from_acc_matrix(acc_matrix::Array{Float32}, k::Int, img_size::Tuple{Int,Int})
    (h, w) = img_size
    top_k = top_k_from_acc_matrix(acc_matrix, k)
    params_list = map(top_k) do (phi_index, r_index, _)
        phi = map_index_to_phi(phi_index, size(acc_matrix)[1])
        r = map_index_to_r(r_index, size(acc_matrix)[2], (w, h)) + eps(Float32)
        (m, n) = phi_and_r_to_m_and_n(phi, r)
        @show phi, r, m, n
        return (m, n)
    end
    return params_list
end

# Helper functions:

function abs_pos_to_rel(xy::Tuple{Int,Int}, wh::Tuple{Int,Int})::Tuple{Int,Int}
    (w, h) = wh
    (x, y) = xy
    return (x - w ÷ 2, (y - h ÷ 2)) # flips the y
end

function rel_pos_to_abs(xy::Tuple{Int,Int}, wh::Tuple{Int,Int})::Tuple{Int,Int}
    (w, h) = wh
    (x, y) = xy
    return (x + w ÷ 2, y + h ÷ 2)
end

function non_maximum_suppression()


"""
expects angle between 0 and pi
yields r between 0 and max()
"""
function rel_pos_and_angle_to_radius(xy::Tuple{Int,Int}, phi::Float64)
    (x, y) = xy
    r = x * cos(phi) + y * sin(phi)
    return r
end

function map_phi_to_index(phi, phi_segements::Int)::Int
    return ceil((phi % pi) / pi * phi_segements)
end

function map_index_to_phi(phi_index, phi_segements::Int)
    return phi_index / phi_segements * pi
end

function map_r_to_index(r, r_segements::Int, wh::Tuple{Int,Int})::Int
    (w, h) = wh
    rmax = sqrt(w^2 + h^2)
    zero_to_one = (r + rmax) / (2 * rmax)
    return ceil(zero_to_one * r_segements)
end

function map_index_to_r(r_index, r_segements::Int, wh::Tuple{Int,Int})
    (w, h) = wh
    rmax = sqrt(w^2 + h^2)
    return (r_index / r_segements * 2 * rmax) - rmax
end

"""
Paramterization with r and phi:
r = x * cos(phi) + y * sin(phi)

Paramterization with m and n:
y = mx + n
"""
function phi_and_r_to_m_and_n(phi::Number, r::Number)::Tuple{Number,Number}
    # finding one point on the line, where x=0:
    x1 = 0 # r = 0 + y * sin(phi)
    y1 = r / sin(phi)

    # finding a second point on the line, where y=0:
    x2 = r / cos(phi) # r = x * cos(phi) + 0
    y2 = 0

    m = (y2 - y1) / (x2 - x1)
    n = y1
    return (m, n)
end
