with source as (
    select * from {{ source('raw', 'raw_stores') }}
),

typed as (
    select
        Store       as store_id,
        Type        as store_type,
        Size        as store_size,
        case Type
            when 'A' then 3
            when 'B' then 2
            else 1
        end         as store_type_num
    from source
)

select * from typed
