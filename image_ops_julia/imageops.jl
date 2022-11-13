using Images
using Colors
using StaticLint
using Test

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
    return gx .+ gy
end

######################################################################################################
######################################################################################################
######################################################################################################

function hugh_transform_accumulator_matrix(edge_img::Matrix{Gray{Float32}}, resolution::Tuple{Int,Int}=(32, 32), threshhold::Float32=Float32(0.8))
    height, width = size(img_grey)
    ϕ_segements, r_segments = resolution
    r_max = sqrt(2) * max(width, height) / 2
    x_center = width ÷ 2
    y_cetner = height ÷ 2
    ϕ_sins::Array{Float32} = map(1:ϕ_segements) do ϕ
        return sin(ϕ / ϕ_segements * 2 * pi)
    end
    ϕ_coss::Array{Float32} = map(1:ϕ_segements) do ϕ
        return cos(ϕ / ϕ_segements * 2 * pi)
    end
    # fill accumulator matrix
    acc_matrix = zeros(Int, resolution)
    maximum::Int = 0
    for x in 1:width
        for y in 1:height
            val = edge_img[y, x]
            if val > threshhold
                x_rel = x - x_center
                y_rel = y - y_cetner
                # formula: x cos(ϕ) + y cos(ϕ) = r
                for ϕ_index in 1:ϕ_segements
                    r = x_rel * ϕ_coss[ϕ_index] + y_rel * ϕ_sins[ϕ_index]
                    r_index = floor(Int, (r + r_max) / (2 * r_max) * r_segments)
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



# tuples represent: Angle phi, Radius r, m, n
function get_top_k_from_acc_matrix(acc_matrix::Array{Float32}, k::Int, img_dim::Tuple{Int,Int})::Array{Tuple{Float32,Float32,Float32,Float32}}

    acc_matrix_dim = size(acc_matrix)
    arr = [(Tuple(ind)[1], Tuple(ind)[2], val) for (ind, val) in pairs(acc_matrix)]
    list = []
    for (y, x, val) in arr
        push!(list, (x, y, val))
    end
    sort!(list, by=x -> x[3])
    top_k = list[length(list)-k+1:end]
    (width, height) = img_dim
    r_max = sqrt(2) * max(width, height) / 2
    return map(top_k) do (p_i, r_i, _)

        # reconstruct phi and r  from indexes
        phi = Float32(p_i) * 2 * pi / acc_matrix_dim[1]
        r = (Float32(r_i) * 2 * r_max / acc_matrix_dim[2]) - r_max
        # finding one point on the line:
        x_rel = sin(phi) * r
        y_rel = cos(phi) * r
        # finding a second point at (0,?):
        # x*cos(phi) + y *sin(phi) = r   =>  if x == 0  =>   y = r / sin(phi) 
        x2_rel = 0
        n = y2_rel = r / sin(phi) # is n in y = mx+n
        m = (y_rel - y2_rel) / (x_rel - x2_rel)

        return (phi, r, m, n)
    end


end





# Helper functions:

function abs_pos_to_rel(xy::Tuple{Int,Int}, wh::Tuple{Int,Int})::Tuple{Int,Int}
    (w, h) = wh
    (x, y) = xy
    return (x - w ÷ 2, y - h ÷ 2)
end

function rel_pos_to_abs(xy::Tuple{Int,Int}, wh::Tuple{Int,Int})::Tuple{Int,Int}
    (w, h) = wh
    (x, y) = xy
    return (x + w ÷ 2, y + h ÷ 2)
end


"""
expects angle between 0 and pi
yields r between 0 and max()
"""
function rel_pos_and_angle_to_radius(xy::Tuple{Int,Int}, phi::Float64)
    (x, y) = xy
    r = x * cos(phi) + y * sin(phi)
    return r
end

function map_phi_to_index(phi, phi_segements::Tuple{Int,Int})::Int
    return ceil((phi % pi) / pi * phi_segements)
end

function map_index_to_phi(phi_index, phi_segements::Tuple{Int,Int})
    return phi_index / phi_segements * pi
end

function map_r_to_index(r, r_segements::Tuple{Int,Int}, wh::Tuple{Int,Int})::Int
    (w, h) = wh
    rmax = sqrt(w^2 + h^2)
    zero_to_one = (r + rmax) / (2 * rmax)
    return ceil(zero_to_one * r_segements)
end

function map_index_to_r(r_index, r_segements::Tuple{Int,Int}, wh::Tuple{Int,Int})::Int
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
