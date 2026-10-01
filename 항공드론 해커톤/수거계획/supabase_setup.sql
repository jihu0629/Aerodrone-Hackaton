-- 문갑도 수거 계획 페이지 · 공유 동기화용 Supabase 설정
-- Supabase 대시보드 → SQL Editor 에 붙여 넣고 실행한다.
--
-- 페이지는 공개용(publishable/anon) 키로 접속하므로 보안은 전적으로 아래 RLS 정책에 달려 있다.
--   · 읽기: 누구나 (링크를 아는 사람 = 현장 팀)
--   · 쓰기: 허용된 site 이름의 행 하나만, 삭제는 아무도 못 함
--   · data 크기 제한으로 큰 쓰레기 값을 넣어 테이블을 망가뜨리는 것을 막는다
--   · 실시간(postgres_changes)을 쓰려면 테이블을 publication 에 넣어야 한다

create table if not exists public.shared_state (
    site        text primary key,
    data        jsonb not null default '{}'::jsonb,
    updated_at  timestamptz not null default now(),
    updated_by  text
);

-- 허용된 조사지 이름만 (새 섬을 추가하면 여기와 아래 정책의 목록을 같이 바꾼다)
alter table public.shared_state
    drop constraint if exists shared_state_site_allowed;
alter table public.shared_state
    add constraint shared_state_site_allowed
    check (site in ('문갑도'));

-- 한 행에 들어가는 JSON 은 200 KB 이하 (완료 체크 수백 개 + 보정 계수면 수 KB 면 충분)
alter table public.shared_state
    drop constraint if exists shared_state_data_size;
alter table public.shared_state
    add constraint shared_state_data_size
    check (pg_column_size(data) < 200000);

-- 이름 칸도 길이 제한
alter table public.shared_state
    drop constraint if exists shared_state_updated_by_len;
alter table public.shared_state
    add constraint shared_state_updated_by_len
    check (updated_by is null or char_length(updated_by) <= 40);

alter table public.shared_state enable row level security;

drop policy if exists "shared_state read"   on public.shared_state;
drop policy if exists "shared_state insert" on public.shared_state;
drop policy if exists "shared_state update" on public.shared_state;

-- 읽기: 누구나
create policy "shared_state read"
    on public.shared_state for select
    to anon, authenticated
    using (true);

-- 쓰기: 허용된 site 행만 (check 제약이 다시 한 번 막는다). delete 정책은 만들지 않는다 → 아무도 못 지움
create policy "shared_state insert"
    on public.shared_state for insert
    to anon, authenticated
    with check (site in ('문갑도'));

create policy "shared_state update"
    on public.shared_state for update
    to anon, authenticated
    using (site in ('문갑도'))
    with check (site in ('문갑도'));

-- 실시간 구독 (페이지가 postgres_changes 로 듣는다). 이미 들어 있으면 오류가 나므로 무시해도 된다.
do $$
begin
    alter publication supabase_realtime add table public.shared_state;
exception when duplicate_object then
    null;
end $$;

-- 확인용
-- select site, updated_at, updated_by, pg_column_size(data) as bytes from public.shared_state;
