module CustomQuad

using QuadGK, DataStructures, LinearAlgebra
import Base.Order.Reverse
import QuadGK.Segment

export quadgk_cauchy, quadgk_custom
export construct_quadrature, segment_quadrature, merge_quadratures
export segment_length

struct RSymInterval{T}
    a::T
    b::T
end

struct RSymSegment{TX, TI, TE}
    rsi::RSymInterval{TX}
    I::TI
    E::TE
end

Base.isless(s1::RSymSegment, s2::RSymSegment) = isless(s1.E, s2.E)
Base.isless(s1::RSymSegment, s2::Segment) = isless(s1.E, s2.E)
Base.isless(s1::Segment, s2::RSymSegment) = isless(s1.E, s2.E)

function get_rs_rule(n::Int64=16)
    x₁, w₁, wg₁ = kronrod(BigFloat, n)
    x₂, w₂, wg₂ = kronrod(BigFloat, 2n)
    x′₁ = Float64.(x₁[2:2:end], RoundNearest)
    wg′₁ = Float64.(wg₁, RoundNearest)
    x′₂ = Float64.(x₂[2:2:end], RoundNearest)
    wg′₂ = Float64.(wg₂, RoundNearest)
    return x′₁, wg′₁, x′₂, wg′₂
end

const rs_rule = get_rs_rule()

function eval_interval(func, rsi::RSymInterval)
    Δ = rsi.b-rsi.a
    b = rsi.b
    I₁ = zero(eltype(b))
    x₁, w₁, x₂, w₂ = rs_rule
    for i in 1:length(x₁)
        y = Δ*x₁[i]
        I₁ += (func(b + y) + func(b - y))*w₁[i]
    end
    I₂ = zero(I₁)
    for i in length(x₂)
        y = Δ*x₂[i]
        I₂ += (func(b + y) + func(b - y))*w₂[i]
    end
    I₁Δ = I₁*Δ
    I₂Δ = I₂*Δ
    return RSymSegment(rsi, I₂Δ, norm(I₂Δ - I₁Δ))
end

eval_interval(func, i::NTuple{2,T}) where {T} = QuadGK.evalrule(func, i[1], i[2], QuadGK.cachedrule(Float64, 7)..., norm)


quadgk_cauchy(func, a, b, c, points...; kws...) = quadgk_cauchy(func, promote(a,b,c, points...)...; kws...)
quadgk_custom(func, a, b, points...; kw...) = quadgk_cauchy(func, promote(a, b, points...)...; kws...)

function process_points_cauchy(a::T, b::T, c::T, points::T...) where {T}
    ordered_points = sort!([a,b, points...])
    m = searchsortedlast(ordered_points, c)
    if c == ordered_points[m]
        l = m - 1 # index of the point to the left of c
    else
        l = m
    end
    r = m + 1     # index of the point to the right of c
    d = min(abs(ordered_points[l] - c)/2, abs(ordered_points[r]-c)/2)
    #######
    intervals = Vector{Union{NTuple{2, T}, RSymInterval}}()
    for i = 1:(l-1)
        push!(intervals, (ordered_points[i], ordered_points[i+1]))
    end
    push!(intervals, (ordered_points[l], c-d))
    push!(intervals, RSymInterval(c-d, c))
    push!(intervals, (c+d, ordered_points[r]))
    for i = r:(length(ordered_points)-1)
        push!(intervals, (ordered_points[i], ordered_points[i+1]))
    end
    return intervals
end

function process_points_custom(a::T, b::T, points::T...) where {T}
    ordered_points = sort!([a,b, points...])
    intervals = Vector{NTuple{2, T}}()
    for i = 1:(length(ordered_points)-1)
        push!(intervals, (ordered_points[i], ordered_points[i+1]))
    end
    return intervals
end

function quadgk_cauchy(func, a::T, b::T, c::T, points::T...; atol = nothing, rtol = nothing, limit = 100) where {T}
    @assert a<c<b
    intervals = process_points_cauchy(a, b, c, points...)
    segs = Union{Segment, RSymSegment}[eval_interval(func, i) for i in intervals]
    I = sum(s -> s.I, segs)
    I′ = sum(s -> norm(s.I), segs)
    E = sum(s -> s.E, segs)

    n_updates = 0
    atol_ = something(atol, zero(E))
    rtol_ = something(rtol, iszero(atol_) ? sqrt(eps(one(T))) : zero(T))

    if (E < atol_) || (E < I′*rtol_)
        return I, E, segs
    end


    heapify!(segs, Reverse)
    while ((E >= atol_) && (E >= I′*rtol_)) && (n_updates<limit)
        update!(segs, func)
        I = sum(s -> s.I, segs)
        I′ = sum(s -> norm(s.I), segs)
        E = sum(s -> s.E, segs)
        n_updates += 1
    end
    if ((E >= atol_) && (E >= I′*rtol_))
        @warn "The algorithm was stopped because the limit on updates was surpassed."
    end
    return I, E, segs
end

function quadgk_custom(func, a::T, b::T, points::T...; atol = nothing, rtol = nothing, limit = 100) where {T}
    @assert a<b
    intervals = process_points_custom(a, b, points...)
    segs = Segment[eval_interval(func, i) for i in intervals]
    I = sum(s -> s.I, segs)
    I′ = sum(s -> norm(s.I), segs)
    E = sum(s -> s.E, segs)

    n_updates = 0
    atol_ = something(atol, zero(E))
    rtol_ = something(rtol, iszero(atol_) ? sqrt(eps(one(T))) : zero(T))

    if (E < atol_) || (E < I′*rtol_)
        return I, E, segs
    end


    heapify!(segs, Reverse)
    while ((E >= atol_) && (E >= I′*rtol_)) && (n_updates<limit)
        update!(segs, func)
        I = sum(s -> s.I, segs)
        I′ = sum(s -> norm(s.I), segs)
        E = sum(s -> s.E, segs)
        n_updates += 1
    end
    if ((E >= atol_) && (E >= I′*rtol_))
        @warn "The algorithm was stopped because the limit on updates was surpassed."
    end
    return I, E, segs
end

function update!(segs::Vector{T}, func) where {T<:Union{Segment, Union{Segment, RSymSegment}}}
    s = heappop!(segs, Reverse)
    new_s = divide(func, s)
    for s′ in new_s
        heappush!(segs, s′, Reverse)
    end
end

@inline function divide(func, s::Segment)
    mid = (s.a + s.b)/2
    s1 = eval_interval(func, (s.a, mid))
    s2 = eval_interval(func, (mid, s.b))
    return s1, s2
end

@inline function divide(func, s::RSymSegment)
    a = s.rsi.a
    b = s.rsi.b
    mid = (a+b)/2
    s1 = eval_interval(func, RSymInterval(mid, b))
    s2 = eval_interval(func, (a, mid))
    s3 = eval_interval(func, ((3b-a)/2, 2b-a))
    return s1, s2, s3
end

function construct_quadrature(segs::Vector{T}) where {T<:Union{Segment, Union{Segment, RSymSegment}}}
    i = findfirst((x)->(typeof(x)==RSymSegment), segs)
    if !isnothing(i)
        temp = segs[i]
        segs[i] = segs[1]
        segs[1] = temp
    end
    return map(segment_quadrature, segs) |> merge_quadratures
end

function segment_quadrature(s::Segment)
    Δ = (s.b-s.a)/2
    mid = (s.b + s.a)/2
    x, w, wg = QuadGK.cachedrule(Float64, 7)
    xg = x[2:2:end]
    x₂ = vcat(x, reverse(-x[1:end-1]))
    w₂ = vcat(w, reverse(w[1:end-1]))
    x₁ = vcat(xg, reverse(-xg[1:end-1]))
    w₁ = vcat(wg, reverse(wg[1:end-1]))
    return x₁.*Δ.+mid, w₁.*Δ, x₂.*Δ.+mid, w₂.*Δ
end

function segment_quadrature(s::RSymSegment)
    x₁, w₁, x₂, w₂ = rs_rule
    rsi = s.rsi
    Δ = rsi.b - rsi.a
    mid = rsi.b
    x₁′ = Vector{Float64}()
    w₁′ = Vector{Float64}()
    x₂′ = Vector{Float64}()
    w₂′ = Vector{Float64}()
    for i = length(x₁):-1:1
        y₁ = x₁[i]*Δ
        v₁ = w₁[i]*Δ
        push!(x₁′, mid+y₁)
        push!(x₁′, mid-y₁)
        append!(w₁′, [v₁, v₁])
    end
    for i = length(x₂):-1:1
        y₂ = x₂[i]*Δ
        v₂ = w₂[i]*Δ
        push!(x₂′, mid+y₂)
        push!(x₂′, mid-y₂)
        append!(w₂′, [v₂, v₂])
    end
    return x₁′, w₁′, x₂′, w₂′
end

function merge_quadratures(xws::Vector{NTuple{4, Vector{Float64}}})
    xa = vcat([xw[1] for xw in xws]...)
    wa = vcat([xw[2] for xw in xws]...)
    x = vcat([xw[3] for xw in xws]...)
    w = vcat([xw[4] for xw in xws]...)
    return xa, wa, x, w
end

segment_length(s::Segment) = (s.b - s.a)
segment_length(s::RSymSegment) = get_length(s.rsi)
get_length(rsi::RSymInterval) = 2*(rsi.b - rsi.a)


end
