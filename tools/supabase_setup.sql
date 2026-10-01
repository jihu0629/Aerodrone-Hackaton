-- ShoreSweep Planner 공유 저장소 (Supabase)
-- Supabase 대시보드 → SQL Editor 에 붙여 넣고 Run.
-- 완료 체크·실측 보정 계수·출발지를 현장(site)마다 한 행(jsonb)으로 저장하고, 모든 접속자가 실시간으로 받는다.

create table if not exists public.shared_state (
  site        text primary key,
  data        jsonb not null default '{}'::jsonb,   -- {done:[...], calib:{...}, depot:{lon,lat,name}}
  updated_at  timestamptz not null default now(),
  updated_by  text
);

-- 행 수준 보안: 링크를 아는 사람(anon 키) 누구나 읽고 쓸 수 있게 (팀 내부 도구용. 더 엄격히 하려면 Supabase Auth 를 붙이고 정책을 바꿀 것)
alter table public.shared_state enable row level security;
drop policy if exists "shared_state read"   on public.shared_state;
drop policy if exists "shared_state insert" on public.shared_state;
drop policy if exists "shared_state update" on public.shared_state;
create policy "shared_state read"   on public.shared_state for select to anon, authenticated using (true);
create policy "shared_state insert" on public.shared_state for insert to anon, authenticated with check (true);
create policy "shared_state update" on public.shared_state for update to anon, authenticated using (true) with check (true);

-- 실시간(postgres_changes) 켜기: 이 테이블의 변경을 구독할 수 있게 publication 에 추가
do $$
begin
  if not exists (select 1 from pg_publication_tables where pubname = 'supabase_realtime' and tablename = 'shared_state') then
    alter publication supabase_realtime add table public.shared_state;
  end if;
end $$;

-- 확인: select * from public.shared_state;
