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

function smoothx(img_grey::Matrix{Gray{Float32}})::Matrix{Gray{Float32}}
    height, width = size(img_grey)
    img2 = zeros(Gray{Float32}, height, width)
    for j in 1:(width)
        for i in 2:(height-1)
            img2[i, j] = (img_grey[i-1, j] + 2 * img_grey[i, j] + img_grey[i+1, j]) / 4
        end
    end
    return img2
end

function harris_corner_detector(img::Matrix{Gray{Float32}}; alpha=0.1, threshold=0.2, min_dist=5)
    is = smooth(img)
    ix = sobelx(is)
    iy = sobely(is)
    A = ix .^ 2
    B = iy .^ 2
    C = ix .* iy
    A̅ = smooth(A)
    B̅ = smooth(B)
    C̅ = smooth(C)
    Q = ((A̅ .* B̅ - (C̅ .^ 2))) .- (alpha * (A̅ + B̅) .^ 2)

    # create corner list:
    Q_suppressed = non_maximum_suppression(Q)
    corners = []
    (h, w) = size(img)
    for x in 1:w
        for y in 1:h
            val = Q_suppressed[y, x]
            if val > threshold
                push!(corners, (x, y, val))
            end
        end
    end
    sort!(corners, by=x -> -x[3])

    # clean up neighbors
    min_sq_dist = min_dist^2
    good_corners = []
    for (x, y) in corners
        isgood = true
        for (x2, y2) in good_corners
            if (x2 - x)^2 + (y2 - y)^2 < min_sq_dist
                isgood = false
                break
            end
        end
        if isgood
            push!(good_corners, (x, y))
        end
    end
    return good_corners
end


function smoothy(img_grey::Matrix{Gray{Float32}})::Matrix{Gray{Float32}}
    height, width = size(img_grey)
    img2 = zeros(Gray{Float32}, height, width)
    for j in 2:(width-1)
        for i in 1:height
            img2[i, j] = (img_grey[i, j-1] + 2 * img_grey[i, j] + img_grey[i, j+1]) / 4
        end
    end
    return img2
end

function smooth(img_grey::Matrix{Gray{Float32}})::Matrix{Gray{Float32}}
    return smoothy(smoothx(img_grey))
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
            if edge_img[y, x] > threshhold
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




function hugh_transform_accumulator_matrix_for_circles(img::Matrix{Gray{Float32}}, resolution::Tuple{Int,Int,Int}=(32, 32, 32); threshhold::Float32=Float32(0.8))
    """
    paramters that specify a circle:
    - center_x
    - center_y
    _ radius

    So we can iterate over each center_x and center_y in a grid and calculate the radius very simply
    """
    acc_matrix = zeros(resolution)
    maximum::Int = 0
    (h, w) = size(img)

    max_circle_size = sqrt(h^2 + w^2)

    for x in 1:w
        for y in 1:h
            if img[y, x] > threshhold
                for cx_index in 1:resolution[1]
                    cx = cx_index * w / resolution[1]
                    for cy_index in 1:resolution[2]
                        cy = cy_index * h / resolution[2]
                        dist = sqrt((x - cx)^2 + (y - cy)^2)
                        dist = clamp(dist, 0, max_circle_size)
                        # @show dist / max_circle_size * resolution[3] - eps(Float32)
                        d_index = ceil(Int, dist / max_circle_size * (resolution[3] - 1) + eps(Float32))
                        acc_matrix[cx_index, cy_index, d_index] += 1
                        if acc_matrix[cx_index, cy_index, d_index] > maximum
                            maximum = acc_matrix[cx_index, cy_index, d_index]
                        end
                    end
                end
            end

        end
    end

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


# tuples represent: phi_index, r_index, value
function top_k_from_3d_acc_matrix(acc_matrix::Array{Float32}, k::Int)::Array{Tuple{Int,Int,Int,Float32}}
    arr = [(Tuple(ind)[1], Tuple(ind)[2], Tuple(ind)[3], val) for (ind, val) in pairs(acc_matrix)]
    list = []
    for (y, x, z, val) in arr
        push!(list, (y, x, z, val))
    end
    sort!(list, by=x -> -x[4])
    top_k = list[1:k]
    return top_k
end

function top_k_circles_from_acc_matrix(acc_matrix::Array{Float32}, k::Int, img_size::Tuple{Int,Int})
    (h, w) = img_size
    top_k = top_k_from_3d_acc_matrix(acc_matrix, k)
    dims = size(acc_matrix)
    max_circle_size = sqrt(h^2 + w^2)
    params_list = map(top_k) do (x_index, y_index, r_index, _)
        x = x_index / dims[1] * w
        y = y_index / dims[2] * h
        r = r_index / dims[3] * max_circle_size
        return (x, y, r)
    end
    return params_list
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

function line_params_from_acc_matrix(acc_matrix::Array{Float32}, threshold::Float32, img_size::Tuple{Int,Int})
    (h, w) = img_size
    arr = [(Tuple(ind)[1], Tuple(ind)[2], val) for (ind, val) in pairs(acc_matrix) if val > threshold]
    params_list = map(arr) do (phi_index, r_index, _)
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

function non_maximum_suppression(matrix::Matrix)
    m2 = copy(matrix)
    (d1, d2) = size(matrix)
    for i in 1:(d1)
        for j in 1:(d2)
            maxi = 0
            for a in (i == 1 ? 0 : -1):(i == d1 ? 0 : 1)
                for b in (j == 1 ? 0 : -1):(j == d2 ? 0 : 1)
                    v = matrix[i+a, j+b]
                    if v > maxi
                        maxi = v
                    end
                end
            end

            if matrix[i, j] != maxi
                m2[i, j] = 0
            end
        end
    end
    return m2
end

function line_params_to_corner_points(params::Vector{Tuple{T,T}}) where {T<:Number}
    """
    For each pair of lines calculate the intersection point
    """
    points = []
    for i in 1:length(params)
        for j in (i+1):length(params)
            # solve:  
            # m1*x+n1 = m2*x+n2
            # m1*x = m2*x+n2-n1
            # m1*x-m2*x = n2-n1
            # (m1-m2)*x = n2-n1
            # x = (n2-n1) / (m1-m2)
            x = (params[j][2] - params[i][2]) / (params[i][1] - params[j][1])
            y = params[i][1] * x + params[i][2]
            push!(points, (x, y))
        end
    end
    return points
end


ex = [(1, 2), (3, 4), (0, 1), (8, 9)]
line_params_to_corner_points(ex)

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

# chamfer matching

function distance_transform(img::Matrix{Gray{Float32}}; m_iterations=3)::Matrix{Float32}
    """
    assumes all entries in img are either 0.0 or 1.0. Only 1.0 values are considered to be edges
    refrence: https://richarddzh.gitbooks.io/book1/content/chamfer_matching.html
    """
    (h, w) = size(img)

    mask = [
        5 4 3 4 5
        4 2 1 2 4
        3 1 0 1 3
        4 2 1 2 4
        5 4 3 4 5
    ]

    # initialize edge_arr
    edge_arr = fill(255, h, w)
    for x in 1:w
        for y in 1:h
            if img[y, x] >= 1.0
                edge_arr[y, x] = 0
            end
        end
    end

    # go for m iterations:
    for _ in 1:m_iterations
        next_edge_arr = copy(edge_arr)
        for x in 3:(w-2)
            for y in 3:(h-2)
                mi = 255
                for i in -2:2
                    for j in -2:2
                        potential_value = edge_arr[y+j, x+i] + mask[i+3, j+3]
                        if mi > potential_value
                            mi = potential_value
                        end
                    end
                end
                next_edge_arr[y, x] = mi
            end
        end
        edge_arr = next_edge_arr
    end


    return edge_arr / 255
end

function chamfer_score(candidate::Matrix{Gray{Float32}}, distance_image::Matrix{Float32})::Float32
    @assert size(candidate) == size(distance_image)
    (h, w) = size(distance_image)
    dist_acc = 0.0
    counter = 0
    for x in 1:w
        for y in 1:h
            if candidate[y, x] >= 1.0
                counter += 1
                dist_acc += distance_image[y, x]
            elseif candidate[y, x] <= 0.0
                counter += 1
                dist_acc -= distance_image[y, x]
            end

        end
    end
    return Float32(dist_acc / counter)
end

function chamfer_matching(haystack::Matrix{Gray{Float32}}, needle::Matrix{Gray{Float32}}; step_size::NTuple{2,Int}=(10, 10), threshold=0.3)::Vector{Tuple{Int,Int,Float32}}
    needle_dist_transform = distance_transform(needle, m_iterations=50)
    step_x, step_y = step_size
    hh, hw = size(haystack)
    nh, nw = size(needle)
    matching_positions::Vector{Tuple{Int,Int,Float32}} = []
    for x in 1:step_x:(hw-nw)
        for y in 1:step_y:(hh-nh)
            slice = haystack[y:y+nh-1, x:x+nw-1]
            @assert size(slice) == size(needle)
            s = chamfer_score(slice, needle_dist_transform)
            push!(matching_positions, (x, y, s))
        end
    end
    filter!(x -> x[3] <= threshold, matching_positions)
    sort!(matching_positions, by=x -> x[3])
    return matching_positions
end
