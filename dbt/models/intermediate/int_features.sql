with sales as (
    select * from {{ ref('stg_sales') }}
),

features as (
    select * from {{ ref('stg_features') }}
),

stores as (
    select * from {{ ref('stg_stores') }}
),

-- Join all sources and compute temporal + context features
joined as (
    select
        s.store_id,
        s.dept_id,
        s.sale_date,
        s.weekly_sales,
        s.is_holiday,

        -- store metadata
        st.store_type,
        st.store_type_num,
        st.store_size,

        -- contextual features
        f.temperature,
        f.fuel_price,
        f.markdown_1,
        f.markdown_2,
        f.markdown_3,
        f.markdown_4,
        f.markdown_5,
        f.cpi,
        f.unemployment,

        -- active markdowns count
        (case when f.markdown_1 > 0 then 1 else 0 end
         + case when f.markdown_2 > 0 then 1 else 0 end
         + case when f.markdown_3 > 0 then 1 else 0 end
         + case when f.markdown_4 > 0 then 1 else 0 end
         + case when f.markdown_5 > 0 then 1 else 0 end
        )                                               as active_markdowns,

        -- temporal features
        extract('week'    from s.sale_date)::integer    as week_of_year,
        extract('month'   from s.sale_date)::integer    as month,
        extract('quarter' from s.sale_date)::integer    as quarter,
        case when extract('month' from s.sale_date) = 12
             then 1 else 0 end                          as is_year_end

    from sales s
    left join stores  st on s.store_id = st.store_id
    left join features f  on s.store_id = f.store_id
                         and s.sale_date = f.sale_date
),

-- Lag features — one row back per store/dept timeline
with_lags as (
    select
        *,
        lag(weekly_sales, 1)  over w as sales_lag_1,
        lag(weekly_sales, 2)  over w as sales_lag_2,
        lag(weekly_sales, 4)  over w as sales_lag_4,
        lag(weekly_sales, 8)  over w as sales_lag_8,
        lag(weekly_sales, 52) over w as sales_lag_52
    from joined
    window w as (partition by store_id, dept_id order by sale_date)
),

-- Rolling aggregations — rows between -w and -1 matches Pandas shift(1).rolling(w)
with_rolling as (
    select
        *,
        avg(weekly_sales)    over w4  as rolling_mean_4,
        stddev(weekly_sales) over w4  as rolling_std_4,
        max(weekly_sales)    over w4  as rolling_max_4,

        avg(weekly_sales)    over w8  as rolling_mean_8,
        stddev(weekly_sales) over w8  as rolling_std_8,
        max(weekly_sales)    over w8  as rolling_max_8,

        avg(weekly_sales)    over w12 as rolling_mean_12,
        stddev(weekly_sales) over w12 as rolling_std_12,
        max(weekly_sales)    over w12 as rolling_max_12

    from with_lags
    window
        w4  as (partition by store_id, dept_id order by sale_date
                rows between 4  preceding and 1 preceding),
        w8  as (partition by store_id, dept_id order by sale_date
                rows between 8  preceding and 1 preceding),
        w12 as (partition by store_id, dept_id order by sale_date
                rows between 12 preceding and 1 preceding)
),

-- Cross-series features — how each store ranks within its dept on a given week
with_cross_series as (
    select
        *,
        avg(weekly_sales) over (
            partition by sale_date, dept_id
        )                                                       as dept_avg_sales_all_stores,

        dense_rank() over (
            partition by sale_date, dept_id
            order by weekly_sales desc
        )                                                       as store_rank_in_dept

    from with_rolling
)

select * from with_cross_series
